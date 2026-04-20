"""Headless Firefox helpers shared by Reddit scraping and generic page fetch.

Reddit's ``.json`` endpoint is fetched inside a rendered page (Selenium) so
that Reddit's bot detection sees a real browser; the same machinery is
reused for article-body fallback when plain HTTP fails (JS-rendered SPAs,
Cloudflare challenges, aggressive UA filters). The helpers here encapsulate
driver construction and optional SOCKS5-proxy auth so callers only deal with
the ``webdriver.Firefox`` instance.

Example::

    driver = build_firefox_driver(timeout_seconds=20.0)
    addon_dir = None
    try:
        if settings.proxy_url:
            addon_dir = create_socks_proxy_addon(settings.proxy_url)
            driver.install_addon(addon_dir, temporary=True)
        driver.get(url)
        html = driver.page_source
    finally:
        driver.quit()
        if addon_dir:
            shutil.rmtree(addon_dir, ignore_errors=True)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from urllib.parse import unquote, urlparse

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"


def build_firefox_driver(timeout_seconds: float) -> webdriver.Firefox:
    """Construct a headless Firefox driver with the shared UA and timeouts."""
    options = Options()
    options.add_argument("-headless")
    options.set_preference("general.useragent.override", BROWSER_USER_AGENT)

    firefox_binary = shutil.which("firefox-esr") or shutil.which("firefox")
    if firefox_binary is not None:
        options.binary_location = firefox_binary

    geckodriver_path = shutil.which("geckodriver")
    service = Service(executable_path=geckodriver_path) if geckodriver_path else Service()
    driver = webdriver.Firefox(service=service, options=options)
    driver.set_page_load_timeout(timeout_seconds)
    driver.set_script_timeout(timeout_seconds)
    return driver


def create_socks_proxy_addon(proxy_url: str) -> str:
    """Build a temporary Firefox extension that routes all traffic via a SOCKS5 proxy.

    Returns the path to the addon directory. Caller is responsible for cleanup
    via ``shutil.rmtree``.
    """
    parsed = urlparse(proxy_url)
    host = parsed.hostname or ""
    port = parsed.port or 1080
    username = unquote(parsed.username) if parsed.username else ""
    password = unquote(parsed.password) if parsed.password else ""

    addon_dir = tempfile.mkdtemp(prefix="firefox_proxy_")

    manifest = {
        "manifest_version": 2,
        "name": "SOCKS5 Proxy Auth",
        "version": "1.0",
        "permissions": ["proxy", "<all_urls>"],
        "background": {"scripts": ["background.js"]},
        "browser_specific_settings": {"gecko": {"id": "proxy-auth@news-service"}},
    }

    background_js = (
        "browser.proxy.onRequest.addListener(\n"
        "  () => ({\n"
        '    type: "socks",\n'
        f"    host: {json.dumps(host)},\n"
        f"    port: {port},\n"
        f"    username: {json.dumps(username)},\n"
        f"    password: {json.dumps(password)},\n"
        "    proxyDNS: true\n"
        "  }),\n"
        '  { urls: ["<all_urls>"] }\n'
        ");\n"
    )

    with open(os.path.join(addon_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(addon_dir, "background.js"), "w") as f:
        f.write(background_js)

    return addon_dir
