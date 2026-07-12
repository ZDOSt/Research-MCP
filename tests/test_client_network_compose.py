from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComposeReset(list):
    pass


class ComposeLoader(yaml.SafeLoader):
    pass


def _construct_compose_reset(loader, node):
    return ComposeReset(loader.construct_sequence(node))


ComposeLoader.add_constructor("!reset", _construct_compose_reset)


class ClientNetworkComposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.override = yaml.load(
            (PROJECT_ROOT / "docker-compose.client-network.yml").read_text("utf-8"),
            Loader=ComposeLoader,
        )
        cls.services = cls.override["services"]

    def test_only_gateway_joins_external_client_network(self):
        attached = [
            name
            for name, service in self.services.items()
            if "client-network" in service.get("networks", {})
        ]
        self.assertEqual(attached, ["mcp-gateway"])
        self.assertTrue(self.override["networks"]["client-network"]["external"])

    def test_gateway_fails_closed_and_uses_configurable_alias(self):
        gateway = self.services["mcp-gateway"]
        self.assertEqual(
            gateway["environment"]["MCP_EXTERNAL_BIND_ADDRESS"], "0.0.0.0"
        )
        self.assertEqual(
            gateway["networks"]["client-network"]["aliases"],
            ["${MCP_CLIENT_ALIAS:-research-mcp}"],
        )
        self.assertIn(
            "${MCP_CLIENT_ALIAS:-research-mcp}:*",
            gateway["environment"]["MCP_ALLOWED_HOSTS"],
        )
        self.assertEqual(
            gateway["environment"]["MCP_ALLOW_UNAUTHENTICATED_REMOTE"], "false"
        )

    def test_gateway_has_no_published_host_ports(self):
        ports = self.services["mcp-gateway"]["ports"]
        self.assertIsInstance(ports, ComposeReset)
        self.assertEqual(ports, [])

    def test_gateway_cannot_resolve_web_backends_on_client_network(self):
        gateway_environment = self.services["mcp-gateway"]["environment"]
        self.assertEqual(gateway_environment["SEARXNG_URL"], "http://127.0.0.1:1")
        self.assertEqual(gateway_environment["CRAWL4AI_URL"], "http://127.0.0.1:1")

    def test_gateway_dependency_names_are_unambiguous(self):
        gateway_environment = self.services["mcp-gateway"]["environment"]
        expected = {
            "redis": ("research-mcp-redis", "REDIS_URL"),
            "qdrant": ("research-mcp-qdrant", "QDRANT_URL"),
            "reranker": ("research-mcp-reranker", "RERANKER_URL"),
        }
        for service_name, (alias, setting) in expected.items():
            with self.subTest(service=service_name):
                aliases = self.services[service_name]["networks"]["backend"][
                    "aliases"
                ]
                self.assertEqual(aliases, [alias])
                self.assertIn(alias, gateway_environment[setting])

    def test_documented_compose_file_retains_client_network_override(self):
        readme = (PROJECT_ROOT / "README.md").read_text("utf-8")
        self.assertIn(
            "COMPOSE_FILE=docker-compose.yml:docker-compose.client-network.yml",
            readme,
        )
        self.assertIn("COMPOSE_PATH_SEPARATOR=:", readme)


if __name__ == "__main__":
    unittest.main()
