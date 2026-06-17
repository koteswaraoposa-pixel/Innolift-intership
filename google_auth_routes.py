import os
import secrets
from datetime import datetime
from urllib.parse import urlencode, urlparse

import requests
import urllib3
from flask import redirect, request, session, url_for, flash

from oauth_helpers import create_user_from_google, get_user_by_google_sub


def _configured_redirect_uri():
    return os.getenv("GOOGLE_REDIRECT_URI")


def _canonical_origin():
    redirect_uri = _configured_redirect_uri()
    if not redirect_uri:
        return None
    parsed = urlparse(redirect_uri)
    return f"{parsed.scheme}://{parsed.netloc}"


def _write_oauth_log(message):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "google_oauth.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def _verify_ssl():
    return os.getenv("GOOGLE_OAUTH_VERIFY_SSL", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _request_kwargs():
    verify = _verify_ssl()
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return {"timeout": 15, "verify": verify}


def register_google_routes(app, oauth):
    @app.route("/auth/google")
    def auth_google():
        if oauth is None:
            flash("Google login is not configured on this server.", "error")
            return redirect(url_for("login"))

        canonical_origin = _canonical_origin()
        if canonical_origin and request.host_url.rstrip("/") != canonical_origin:
            return redirect(f"{canonical_origin}{url_for('auth_google')}")

        redirect_uri = _configured_redirect_uri() or url_for("auth_google_callback", _external=True)
        session["oauth_next"] = request.args.get("next") or url_for("predict")

        client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not client_id:
            flash("Google OAuth is not configured correctly (client missing).", "error")
            return redirect(url_for("login"))

        state = secrets.token_urlsafe(32)
        session["google_oauth_state"] = state
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "email profile",
            "state": state,
            "prompt": "select_account",
        }
        return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


    @app.route("/auth/google/callback")
    def auth_google_callback():
        if oauth is None:
            flash("Google login is not configured on this server.", "error")
            return redirect(url_for("login"))

        if request.args.get("error"):
            error = request.args.get("error")
            description = request.args.get("error_description", "")
            _write_oauth_log(f"google_returned_error: {error}: {description}")
            flash("Google login was cancelled or rejected.", "error")
            return redirect(url_for("login"))

        expected_state = session.pop("google_oauth_state", None)
        returned_state = request.args.get("state")
        if not expected_state or expected_state != returned_state:
            _write_oauth_log("callback_failed: OAuth state mismatch")
            flash("Google login session expired. Please try again.", "error")
            return redirect(url_for("login"))

        code = request.args.get("code")
        if not code:
            _write_oauth_log("callback_failed: missing authorization code")
            flash("Google login failed. Please try again.", "error")
            return redirect(url_for("login"))

        redirect_uri = _configured_redirect_uri() or url_for("auth_google_callback", _external=True)
        try:
            token_response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                **_request_kwargs(),
            )
            if not token_response.ok:
                _write_oauth_log(
                    f"token_failed: {token_response.status_code}: {token_response.text[:500]}"
                )
            token_response.raise_for_status()
            token = token_response.json()

            userinfo_response = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token['access_token']}"},
                **_request_kwargs(),
            )
            if not userinfo_response.ok:
                _write_oauth_log(
                    f"userinfo_failed: {userinfo_response.status_code}: {userinfo_response.text[:500]}"
                )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
        except Exception as exc:
            app.logger.exception("Google OAuth callback failed")
            _write_oauth_log(f"callback_failed: {type(exc).__name__}: {exc}")
            flash("Google login failed. Please try again.", "error")
            return redirect(url_for("login"))

        google_sub = userinfo.get("sub") or userinfo.get("id")
        email = userinfo.get("email")
        name = userinfo.get("name")

        if not google_sub:
            flash("Google login failed: missing Google id.", "error")
            return redirect(url_for("login"))

        db_user = get_user_by_google_sub(google_sub)
        if db_user is None:
            user_id = create_user_from_google(google_sub, email, name)
        else:
            user_id = db_user["id"]

        session["user_id"] = user_id
        flash("Logged in with Google.", "success")

        next_url = session.pop("oauth_next", None) or url_for("predict")
        return redirect(next_url)

