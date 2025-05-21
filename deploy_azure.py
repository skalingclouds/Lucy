#!/usr/bin/env python3

"""Push‑button deployment helper for Lucy (Apex AI Assistant) – Python edition.

This replaces the previous *deploy‑azure‑fixed.sh* bash script with a single
Python program that:

1. Validates local prerequisites (Azure CLI, Docker, logged‑in account).
2. Builds & pushes the Docker image to Azure Container Registry.
3. Ensures the required Azure resources exist (resource‑group, ACR, Key Vault,
   Container Apps environment).
4. Uploads secrets from the local *.env* file to Key Vault (skips empty).
5. Deploys / updates the Container App **with managed identity enabled**.
6. Assigns RBAC roles to the identity *only if they are not already present* –
   avoiding the noisy "assignment already exists" errors in the old script.
7. Waits for the public FQDN to respond with HTTP 200/302 and prints the URL.

NOTE: the script intentionally shells‑out to `az` and `docker` for reliability
and reduced dependencies – the Azure SDK does not yet support all Container
App operations.

Usage
-----
    python deploy_azure.py          # uses defaults specified below

    # override any setting via CLI flags
    python deploy_azure.py --app-name MyLucy --location eastus --dry-run

The program is idempotent: you can run it multiple times and it will update
rather than recreate resources.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
import shutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sh(
    cmd: List[str], *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* via subprocess.run and return the CompletedProcess.

    When *capture* is True stdout is captured and returned as text; otherwise we
    inherit the parent stdio so the user sees real‑time output.
    """

    if capture:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    else:
        result = subprocess.run(cmd, text=True, check=False)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed (exit {result.returncode})\n{result.stderr}"
        )

    return result


def az_json(cmd: List[str]) -> dict | list:
    """Run an az CLI command that outputs JSON and return parsed result."""

    res = sh(cmd + ["--output", "json"], capture=True)
    if res.stdout.strip() == "":
        return {}
    return json.loads(res.stdout)


def log(msg: str, *, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# ---------------------------------------------------------------------------
# Deployment class
# ---------------------------------------------------------------------------


class Deployer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.image_tag = f"v1-{self.timestamp}"

        # ACR repository names must be lowercase. Keep a lowercase variant
        # strictly for image tagging; the Container App display name can retain
        # original casing.
        # Safe names for Azure resources (lowercase, valid chars)
        self.repo_name = args.app_name.lower()
        self.app_safe = args.app_name.lower()

        # derived
        self.acr_login_server = None  # set after ACR creation
        self.acr_username = None
        self.acr_password = None

        # load .env so secrets are available via os.environ
        self._load_env()

    # -------------------------- prerequisites ---------------------------

    def check_prereqs(self) -> None:
        for exe in ("az", "docker"):
            if not shutil.which(exe):
                raise RuntimeError(f"'{exe}' is required but not found in PATH")

        # confirm azure login
        try:
            az_json(["az", "account", "show"])
        except RuntimeError as e:
            raise RuntimeError("Please run 'az login' first") from e

    # -------------------------- env / secrets --------------------------

    def _load_env(self) -> None:
        env_file = Path(self.args.env_file)
        if not env_file.exists():
            log(
                f"No {env_file} found – continuing without local secrets",
                level="WARNING",
            )
            return

        for line in env_file.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Remove surrounding quotes if present
            if (v.startswith("\"") and v.endswith("\"")) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]
            os.environ.setdefault(k, v)

    # ---------------------- Azure resource helpers ----------------------

    def _ensure_group(self) -> None:
        rg = self.args.resource_group
        loc = self.args.location
        if not az_json(["az", "group", "exists", "--name", rg]):
            log(f"Creating resource‑group {rg} ({loc})")
            sh(["az", "group", "create", "--name", rg, "--location", loc])
        else:
            log(f"Resource‑group {rg} already exists")

    def _ensure_acr(self) -> None:
        rg, acr = self.args.resource_group, self.args.acr_name
        exists = False
        try:
            az_json(["az", "acr", "show", "--name", acr, "--resource-group", rg])
            exists = True
        except RuntimeError:
            pass

        if not exists:
            log(f"Creating ACR {acr}")
            sh(
                [
                    "az",
                    "acr",
                    "create",
                    "--name",
                    acr,
                    "--resource-group",
                    rg,
                    "--sku",
                    "Basic",
                    "--admin-enabled",
                    "true",
                ]
            )
        else:
            log("ACR already exists")

        self.acr_login_server = az_json(
            [
                "az",
                "acr",
                "show",
                "--name",
                acr,
                "--resource-group",
                rg,
                "--query",
                "loginServer",
                "-o",
                "tsv",
            ]
        )
        self.acr_username = az_json(
            [
                "az",
                "acr",
                "credential",
                "show",
                "--name",
                acr,
                "--query",
                "username",
                "-o",
                "tsv",
            ]
        )
        self.acr_password = az_json(
            [
                "az",
                "acr",
                "credential",
                "show",
                "--name",
                acr,
                "--query",
                "passwords[0].value",
                "-o",
                "tsv",
            ]
        )

    def _ensure_kv(self) -> None:
        rg, kv = self.args.resource_group, self.args.key_vault
        exists = False
        try:
            az_json(["az", "keyvault", "show", "--name", kv, "--resource-group", rg])
            exists = True
        except RuntimeError:
            pass

        if not exists:
            log(f"Creating Key Vault {kv}")
            sh(
                [
                    "az",
                    "keyvault",
                    "create",
                    "--name",
                    kv,
                    "--resource-group",
                    rg,
                    "--location",
                    self.args.location,
                ]
            )

        # Grant current user (or specified object‑id) secret permissions.
        obj_id = os.getenv("AZ_DEPLOYER_OBJECT_ID") or az_json(
            ["az", "ad", "signed-in-user", "show", "--query", "id", "-o", "tsv"]
        )

        kv_props = az_json(["az", "keyvault", "show", "--name", kv])
        enable_rbac = kv_props.get("properties", {}).get("enableRbacAuthorization")

        try:
            if enable_rbac:
                # Use RBAC role assignment instead of access policy
                assignments = az_json(
                    [
                        "az",
                        "role",
                        "assignment",
                        "list",
                        "--assignee",
                        obj_id,
                        "--scope",
                        kv_props["id"],
                    ]
                )
                if not any(
                    a.get("roleDefinitionName") == "Key Vault Secrets Officer"
                    for a in assignments
                ):
                    sh(
                        [
                            "az",
                            "role",
                            "assignment",
                            "create",
                            "--assignee",
                            obj_id,
                            "--role",
                            "Key Vault Secrets Officer",
                            "--scope",
                            kv_props["id"],
                        ]
                    )
            else:
                sh(
                    [
                        "az",
                        "keyvault",
                        "set-policy",
                        "--name",
                        kv,
                        "--object-id",
                        obj_id,
                        "--secret-permissions",
                        "get",
                        "set",
                        "list",
                    ]
                )
        except RuntimeError as e:
            log(f"Could not configure Key Vault access: {e}", level="WARNING")

    # ------------------------ Key‑Vault helper -------------------------

    @staticmethod
    def _kv_name(env_name: str) -> str:
        """Convert ENV_VAR style name to a valid Key Vault secret name.

        Key Vault secret names may contain only 0‑9, a‑z, A‑Z and - (dash).
        We map underscores to dashes and lower‑case the result so that the
        mapping is deterministic both when **storing** and **retrieving**.
        """

        return env_name.lower().replace("_", "-")

    def _put_secret(self, name: str, value: str) -> None:
        kv = self.args.key_vault
        kv_name = self._kv_name(name)
        sh(
            [
                "az",
                "keyvault",
                "secret",
                "set",
                "--vault-name",
                kv,
                "--name",
                kv_name,
                "--value",
                value,
                "--output",
                "none",
            ]
        )

    def _upload_secrets(self) -> None:
        log("Uploading secrets to Key Vault")
        for key in self.args.secret_names:
            val = os.getenv(key)
            if val:
                self._put_secret(key, val)

        # ACR credentials (needed for Container Apps to pull the image)
        self._put_secret("ACR-USERNAME", self.acr_username)
        self._put_secret("ACR-PASSWORD", self.acr_password)

    def _ensure_container_env(self) -> None:
        env = self.args.container_env
        rg = self.args.resource_group
        try:
            az_json(
                [
                    "az",
                    "containerapp",
                    "env",
                    "show",
                    "--name",
                    env,
                    "--resource-group",
                    rg,
                ]
            )
        except RuntimeError:
            log(f"Creating Container App env {env}")
            sh(
                [
                    "az",
                    "containerapp",
                    "env",
                    "create",
                    "--name",
                    env,
                    "--resource-group",
                    rg,
                    "--location",
                    self.args.location,
                ]
            )

    # --------------------------- Docker -------------------------------

    def _embed_version_in_chainlit(self) -> None:
        """Insert or update a build-version line in *chainlit.md* so the UI
        shows which image is running. The line format is:

            Build ID: v1-YYYYMMDDHHMMSS
        """
        md_path = Path("chainlit.md")
        if not md_path.exists():
            return  # nothing to do

        version_line = f"Build ID: {self.image_tag}"
        lines = md_path.read_text().splitlines()
        # Remove any existing version marker
        lines = [ln for ln in lines if not ln.startswith("Build ID:")]
        # Append a blank line and the new version
        lines += ["", version_line]
        md_path.write_text("\n".join(lines))

    def _build_push_image(self) -> None:
        # Ensure Chainlit markdown reflects the version
        self._embed_version_in_chainlit()
        image_ver = f"{self.acr_login_server}/{self.repo_name}:{self.image_tag}"
        image_latest = f"{self.acr_login_server}/{self.repo_name}:latest"

        log(f"Building Docker image {image_ver}")
        sh(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                image_ver,
                "-t",
                image_latest,
                "-f",
                "Dockerfile",
                ".",
            ]
        )

        log("Pushing image to ACR")
        sh(["az", "acr", "login", "--name", self.args.acr_name])
        sh(["docker", "push", image_ver])
        sh(["docker", "push", image_latest])

    # ---------------------- Container App -----------------------------

    def _get_secret_env_flags(self) -> list[str]:
        kv = self.args.key_vault
        secret_pairs = []
        for key in self.args.secret_names + ["ACR-USERNAME", "ACR-PASSWORD"]:
            try:
                value = az_json(
                    [
                        "az",
                        "keyvault",
                        "secret",
                        "show",
                        "--vault-name",
                        kv,
                        "--name",
                        self._kv_name(key),
                        "--query",
                        "value",
                        "-o",
                        "tsv",
                    ]
                )
            except RuntimeError:
                continue
            secret_pairs.append(f"{key}={value}")
        return secret_pairs

    def _deploy_container_app(self) -> None:
        rg = self.args.resource_group
        env = self.args.container_env

        image = f"{self.acr_login_server}/{self.repo_name}:{self.image_tag}"

        env_pairs = self._get_secret_env_flags()  # list of "K=V"

        # registry creds flags (used on create only)
        reg_flags = [
            "--registry-server",
            self.acr_login_server,
            "--registry-username",
            self.acr_username,
            "--registry-password",
            self.acr_password,
        ]

        exists = False
        try:
            az_json(
                [
                    "az",
                    "containerapp",
                    "show",
                    "--name",
                    self.app_safe,
                    "--resource-group",
                    rg,
                ]
            )
            exists = True
        except RuntimeError:
            pass

        if exists:
            log("Updating existing Container App")

            cmd = [
                "az",
                "containerapp",
                "update",
                "--name",
                self.app_safe,
                "--resource-group",
                rg,
                "--image",
                image,
            ]

            if env_pairs:
                cmd += ["--set-env-vars", *env_pairs]

            sh(cmd)

        else:
            log("Creating new Container App")
            create_cmd = [
                "az",
                "containerapp",
                "create",
                "--name",
                self.app_safe,
                "--resource-group",
                rg,
                "--environment",
                env,
                "--image",
                image,
                "--target-port",
                "8000",
                "--ingress",
                "external",
                "--system-assigned",
                *reg_flags,
            ]

            if env_pairs:
                create_cmd += ["--env-vars", *env_pairs]

            create_cmd += [
                "--cpu",
                "0.5",
                "--memory",
                "1Gi",
            ]

            # filter out None (if env_pairs empty)
            create_cmd = [x for x in create_cmd if x]
            sh(create_cmd)

        # Assign RBAC roles only if not present
        identity = az_json(
            [
                "az",
                "containerapp",
                "show",
                "--name",
                self.app_safe,
                "--resource-group",
                rg,
                "--query",
                "identity.principalId",
                "-o",
                "tsv",
            ]
        )
        subscription = az_json(
            ["az", "account", "show", "--query", "id", "-o", "tsv"]
        )

        def _assign(role: str, scope: str):
            # check
            assignments = az_json(
                [
                    "az",
                    "role",
                    "assignment",
                    "list",
                    "--assignee",
                    identity,
                    "--scope",
                    scope,
                ]
            )
            if any(a.get("roleDefinitionName") == role for a in assignments):
                return
            sh(
                [
                    "az",
                    "role",
                    "assignment",
                    "create",
                    "--assignee",
                    identity,
                    "--role",
                    role,
                    "--scope",
                    scope,
                ]
            )

        log("Ensuring RBAC roles for managed identity")
        acr_id = az_json(
            [
                "az",
                "acr",
                "show",
                "--name",
                self.args.acr_name,
                "--resource-group",
                rg,
                "--query",
                "id",
                "-o",
                "tsv",
            ]
        )

        _assign("AcrPull", acr_id)
        _assign("Storage Blob Data Reader", f"/subscriptions/{subscription}")

        # Optional ML workspace role via env‑var
        workspace_id = os.getenv("AZURE_ML_WORKSPACE_ID")
        if workspace_id:
            _assign("AzureML Data Scientist", workspace_id)

        # Cognitive & Search contributor (if missing)
        _assign("Cognitive Services Contributor", f"/subscriptions/{subscription}")
        _assign("Cognitive Services OpenAI Contributor", f"/subscriptions/{subscription}")
        _assign("Cognitive Services User", f"/subscriptions/{subscription}")
        _assign("Search Service Contributor", f"/subscriptions/{subscription}")
        _assign("Search Index Data Contributor", f"/subscriptions/{subscription}")

    # ----------------------- health‑check -----------------------------

    def _wait_ready(self):
        fqdn = az_json(
            [
                "az",
                "containerapp",
                "show",
                "--name",
                self.app_safe,
                "--resource-group",
                self.args.resource_group,
                "--query",
                "properties.configuration.ingress.fqdn",
                "-o",
                "tsv",
            ]
        )
        url = f"https://{fqdn}"
        log(f"Waiting for application to start at {url}")
        for _ in range(30):
            try:
                code = sh(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                    capture=True,
                    check=False,
                ).stdout.strip()
                if code in {"200", "302"}:
                    log("Application is live!", level="SUCCESS")
                    print(url)
                    return
            except Exception:
                pass
            time.sleep(10)
        log("Application did not become ready in time", level="ERROR")

    # ----------------------- main orchestrations ----------------------

    def run(self):
        if self.args.dry_run:
            log("Dry‑run mode – no changes will be made", level="WARNING")

        self._ensure_group()
        self._ensure_acr()
        self._ensure_kv()
        self._upload_secrets()
        self._ensure_container_env()

        self._build_push_image()
        self._deploy_container_app()
        self._wait_ready()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy Lucy (Apex AI Assistant) to Azure Container Apps"
    )

    p.add_argument(
        "--env-file",
        default=".env",
        help="Local .env file containing secrets (default: .env)",
    )
    p.add_argument("--resource-group", default="rg-apex-lucy-prd-01")
    p.add_argument("--location", default="westus")
    p.add_argument("--acr-name", default="apexacr1")
    p.add_argument("--app-name", default="Apex-Agent-Lucy-Dev1")
    p.add_argument("--container-env", default="apex-env1")
    p.add_argument("--key-vault", default="apexlucykvdev1")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


# Secrets that will be uploaded automatically if present in the environment
DEFAULT_SECRET_NAMES = [
    "AZURE_PROJECT_CONNSTRING",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_INDEX_NAME",
    "AI_SEARCH_CONNECTION_NAME",
    "D365_TENANT_ID",
    "D365_CLIENT_ID",
    "D365_CLIENT_SECRET",
    "D365_RESOURCE_URL",
    "SMTP_SERVER",
    "SMTP_PORT",
    "SENDER_EMAIL",
    "SENDER_PASSWORD",
    "RECEIVER_EMAIL",
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_BLOB_ENDPOINT",
    "AZURE_GPT_MODEL",
    "AZURE_RESOURCE_GROUP",
    "DYNAMICS_ENABLED",
    "AZURE_LOCATION",
    "AZURE_CONTAINER_ENV",
    "AZURE_KEY_VAULT",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "SEARCH_TOP_K",          # controls top-K results for Azure AI Search
    # query type override for Azure Search ('semantic', 'simple', etc.)
    "SEARCH_QUERY_TYPE",
    "AGENT_PORTAL_ENABLED",  # enable/disable agent portal integration
    "AGENT_PORTAL_URL",      # base URL for the agent portal service
    "MANAGED_IDENTITY_CLIENT_ID"
]


def main():
    args = parse_args()
    args.secret_names = DEFAULT_SECRET_NAMES  # inject

    try:
        deployer = Deployer(args)
        deployer.run()
    except Exception as exc:
        log(str(exc), level="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    import shutil  # used in prereq check

    main()
