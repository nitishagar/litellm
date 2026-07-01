import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from litellm.proxy.hooks.user_management_event_hooks import (
    UserManagementEventHooks,
    _should_send_user_invitation_email,
)
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy._types import (
    NewUserRequest,
    NewUserResponse,
    GenerateKeyRequest,
    GenerateKeyResponse,
    UserAPIKeyAuth,
)
import builtins
import sys
from types import SimpleNamespace


@pytest.mark.asyncio
async def test_v1_user_creation_no_email_when_send_invite_email_false():
    """
    Test that user invitation email is NOT sent when send_invite_email=False
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch(
        "litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]
    ):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=False,  # Should NOT send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(
                user_id="admin-user", api_key="admin-key"
            )
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_v1_user_creation_sends_email_when_send_invite_email_true():
    """
    Test that user invitation email IS sent when send_invite_email=True
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch(
        "litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]
    ):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=True,  # Should send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(
                user_id="admin-user", api_key="admin-key"
            )
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_called_once()


@pytest.mark.asyncio
async def test_v1_key_generation_sends_email_when_send_invite_email_true():
    """
    Test that key generation email IS sent when send_invite_email=True
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch.object(
        KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email
    ):
        with patch(
            "litellm.logging_callback_manager.get_custom_loggers_for_type",
            return_value=[],
        ):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(
                sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}
            ):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=True,  # Should send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(
                    user_id="admin-user", api_key="admin-key"
                )
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_called_once()


@pytest.mark.asyncio
async def test_v1_key_generation_no_email_when_send_invite_email_false():
    """
    Test that key generation email is NOT sent when send_invite_email=False
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch.object(
        KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email
    ):
        with patch(
            "litellm.logging_callback_manager.get_custom_loggers_for_type",
            return_value=[],
        ):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(
                sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}
            ):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=False,  # Should NOT send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(
                    user_id="admin-user", api_key="admin-key"
                )
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_not_called()


# ---------------------------------------------------------------------------
# Helper unit tests (I1, I2, I3, I4)
# ---------------------------------------------------------------------------


def test_should_send_when_none_and_email_configured_and_user_has_email():
    assert _should_send_user_invitation_email(None, email_configured=True, user_email="user@example.com") is True


def test_should_not_send_when_none_and_email_not_configured():
    assert _should_send_user_invitation_email(None, email_configured=False, user_email="user@example.com") is False


def test_should_not_send_when_none_and_user_email_missing():
    assert _should_send_user_invitation_email(None, email_configured=True, user_email=None) is False


def test_should_not_send_when_none_and_user_email_empty():
    assert _should_send_user_invitation_email(None, email_configured=True, user_email="") is False


def test_should_send_when_true_regardless_of_config():
    assert _should_send_user_invitation_email(True, email_configured=False, user_email=None) is True


def test_should_not_send_when_false_regardless_of_config():
    assert _should_send_user_invitation_email(False, email_configured=True, user_email="user@example.com") is False


# ---------------------------------------------------------------------------
# Integration tests for async_send_user_invitation_email
# ---------------------------------------------------------------------------


def _make_legacy_patch_context(alerting_configured: bool):
    mock_slack = MagicMock()
    mock_slack.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.slack_alerting_instance = mock_slack
    alerting = ["email"] if alerting_configured else []
    mock_proxy_server = SimpleNamespace(
        general_settings={"alerting": alerting},
        proxy_logging_obj=mock_proxy_logging,
        litellm_proxy_admin_name="admin-user",
    )
    return mock_proxy_server, mock_slack


@pytest.mark.asyncio
async def test_v1_user_creation_sends_email_when_send_invite_email_omitted_and_email_configured():
    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=True)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="user@example.com")
            response = NewUserResponse(user_id="u1", user_email="user@example.com", key="sk-k")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_slack.send_key_created_or_user_invited_email.assert_called_once()


@pytest.mark.asyncio
async def test_v1_user_creation_no_email_when_send_invite_email_omitted_and_email_not_configured():
    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=False)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="user@example.com")
            response = NewUserResponse(user_id="u1", user_email="user@example.com", key="sk-k")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_no_send_when_user_email_missing():
    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=True)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest()
            response = NewUserResponse(user_id="u1", user_email=None, key="sk-k")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_user_creation_sends_enterprise_email_when_send_invite_email_omitted():
    try:
        from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
    except ImportError:
        pytest.skip("litellm_enterprise not installed")

    mock_enterprise_logger = MagicMock()
    mock_enterprise_logger.__class__ = BaseEmailLogger
    mock_enterprise_logger.send_user_invitation_email = AsyncMock()

    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=False)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[mock_enterprise_logger]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="user@example.com")
            response = NewUserResponse(user_id="u1", user_email="user@example.com", key="sk-k")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_enterprise_logger.send_user_invitation_email.assert_called_once()
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_no_double_send_when_enterprise_and_legacy_both_configured():
    try:
        from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
    except ImportError:
        pytest.skip("litellm_enterprise not installed")

    mock_enterprise_logger = MagicMock()
    mock_enterprise_logger.__class__ = BaseEmailLogger
    mock_enterprise_logger.send_user_invitation_email = AsyncMock()

    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=True)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[mock_enterprise_logger]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="user@example.com")
            response = NewUserResponse(user_id="u1", user_email="user@example.com", key="sk-k")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_enterprise_logger.send_user_invitation_email.assert_called_once()
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_sso_user_creation_does_not_send_invite_email():
    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=True)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="sso-user@example.com", send_invite_email=False)
            response = NewUserResponse(user_id="sso-u1", user_email="sso-user@example.com", key="sk-sso")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_scim_user_creation_does_not_send_invite_email():
    mock_proxy_server, mock_slack = _make_legacy_patch_context(alerting_configured=True)
    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(user_email="scim-user-id", send_invite_email=False)
            response = NewUserResponse(user_id="scim-u1", user_email="scim-user-id", key="sk-scim")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data, response=response, user_api_key_dict=UserAPIKeyAuth(user_id="admin")
            )
            mock_slack.send_key_created_or_user_invited_email.assert_not_called()
