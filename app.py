# app.py
"""
Clinician-facing Streamlit UI.

Identity comes from a trusted authenticating reverse proxy (e.g. oauth2-proxy in front of
your IdP) that sets X-Forwarded-User / X-Forwarded-Email / X-Forwarded-Groups. Streamlit
MUST NOT be reachable except through that proxy, otherwise these headers can be forged.
Clearance groups shown to the user are the intersection of their IdP groups and
ALLOWED_CLEARANCE_GROUPS -- users can never type in an arbitrary group.
"""
from __future__ import annotations

import logging
import os

import streamlit as st

from config import configure_logging, get_settings
from main_pipeline import execute_clinical_query

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Clinical Reference Assistant", layout="centered")


def resolve_identity() -> tuple[str | None, list[str]]:
    s = get_settings()
    headers = {k.lower(): v for k, v in st.context.headers.items()}
    user = headers.get("x-forwarded-email") or headers.get("x-forwarded-user")
    groups_raw = headers.get("x-forwarded-groups", "")
    if not user and s.environment == "development":
        user = os.getenv("DEV_USER", "dev-user")
        groups_raw = os.getenv("DEV_GROUPS", "")
    groups = sorted({g.strip().lower() for g in groups_raw.split(",") if g.strip()} & s.allowed_groups)
    return user, groups


user_id, groups = resolve_identity()

st.title("Clinical Reference Assistant")
st.caption(
    "Answers are generated only from approved internal guidelines and must be verified by a licensed "
    "clinician before use. Do not enter patient-identifying information."
)

if not user_id:
    st.error("Not authenticated. Access this application through the organisation's SSO gateway.")
    st.stop()
if not groups:
    st.error("Your account is not assigned to any clinical reference group. Contact your administrator.")
    st.stop()

group = groups[0] if len(groups) == 1 else st.selectbox("Reference scope", groups)

with st.form("ask", clear_on_submit=False):
    question = st.text_area("Clinical question", max_chars=get_settings().max_question_chars, height=120)
    submitted = st.form_submit_button("Ask")

if submitted:
    try:
        with st.spinner("Searching approved guidelines..."):
            result = execute_clinical_query({"user_id": user_id, "clearance_group": group}, question)
    except (ValueError, PermissionError) as exc:
        st.warning(str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("query failed")
        st.error("The assistant is temporarily unavailable. Please try again or consult the source guideline.")
    else:
        if result.grounded:
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Sources"):
                    for src in result.sources:
                        st.write(f"[{src['id']}] {src['source_document']} - page {src['page_number']}")
        else:
            st.info(result.answer)
        st.caption(f"Reference ID: {result.request_id}" + (" - served from cache" if result.cached else ""))
