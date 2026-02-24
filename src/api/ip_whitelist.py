# src/api/ip_whitelist.py
import yaml
from fastapi import Request
from ipaddress import ip_address
from src.api.response import ApiResponse
from fastapi.responses import JSONResponse

class IPWhitelistMiddleware:
    def __init__(self, app):
        self.app = app
        with open('config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            self.auth = cfg.get("auth", {})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        client_ip = ip_address(request.client.host)

        if not self.auth.get("enabled", False):
            await self.app(scope, receive, send)
            return

        allowed_ips = self.auth.get("allowed_ips", [])
        allowed = [ip_address(ip) for ip in allowed_ips]

        if client_ip not in allowed:
            response = JSONResponse(
                status_code=404,
                content=ApiResponse.fail("IP地址未授权", code=404).dict()
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
