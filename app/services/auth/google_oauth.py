import base64
import hashlib
import hmac
import secrets
from typing import List

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
import google.auth.transport.requests

# Use full scopes to avoid "scope changed" warnings
SCOPES: List[str] = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")

def _make_state(secret: str) -> str:
    # random nonce + HMAC(secret, nonce)
    nonce = secrets.token_urlsafe(16)
    sig = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).digest()
    return f"{nonce}.{_b64url(sig)}"

def _verify_state(state: str, secret: str) -> bool:
    try:
        nonce, sig_b64 = state.split(".", 1)
        expected = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).digest()
        return hmac.compare_digest(sig_b64, _b64url(expected))
    except Exception:
        return False

def _qp(params, key: str) -> str:
    v = params.get(key, "")
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""

def require_google_auth() -> str:
    """Authenticate via Google OAuth and return the user's email."""
    client_id = st.secrets.get("GOOGLE_CLIENT_ID")
    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = st.secrets.get("GOOGLE_REDIRECT_URI")
    allowed = st.secrets.get("ALLOWED_EMAILS", [])

    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    if not client_id or not client_secret:
        st.error("Google OAuth is not configured.")
        st.stop()

    params = st.query_params
    code = _qp(params, "code")
    state = _qp(params, "state")

    # 1) Start auth
    if not code:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
            redirect_uri=redirect_uri,
        )
        signed_state = _make_state(client_secret)
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
            state=signed_state,
        )
        button_html = f"""
        <a href="{auth_url}" target="_self" style="text-decoration:none;">
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                background-color:white;
                color:#444;
                border:1px solid #dadce0;
                border-radius:4px;
                padding:0.5em 1em;
                font-size:16px;
                font-weight:500;
                cursor:pointer;">
                <img src="https://developers.google.com/identity/images/g-logo.png" style="height:18px;margin-right:8px;">
                <span>Sign in with Google</span>
            </div>
        </a>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        st.stop()

    # 2) Callback: verify state
    if not _verify_state(state, client_secret):
        st.error("State mismatch.")
        st.stop()

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)

    request = google.auth.transport.requests.Request()
    info = id_token.verify_oauth2_token(flow.credentials.id_token, request, client_id)
    email = info.get("email")

    if allowed and email not in allowed:
        st.warning("Unauthorized email.")
        st.stop()

    st.session_state["user_email"] = email
    st.query_params.clear()
    return email


def logout() -> None:
    """Clear stored user session and rerun the app."""
    st.session_state.pop("user_email", None)
    st.rerun()
