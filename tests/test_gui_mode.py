"""
Tests for GUI mode feature.

Run with: python -m pytest tests/test_gui_mode.py

Tests cover:
- Headless flag presence/absence in JVM startup
- Autoplot agent prompt GUI-mode content
- Tool schema count (14 total)
- AutoplotCommands.reset() clears internal state
- Singleton guard for conflicting gui_mode
- waitUntilIdle() calls on DOM-mutating operations
"""

import pytest
from unittest.mock import patch, MagicMock, call


class TestConnectionHeadless:
    """Test that the headless flag is correctly passed to JVM startup."""

    @patch("autoplot_bridge.connection.jpype")
    @patch("autoplot_bridge.connection.AUTOPLOT_JAR", "fake.jar")
    @patch("autoplot_bridge.connection.JAVA_HOME", None)
    def test_headless_true_passes_flag(self, mock_jpype):
        """Default headless=True should pass -Djava.awt.headless=true."""
        # Reset module state
        import autoplot_bridge.connection as conn
        conn._headless_mode = True

        mock_jpype.isJVMStarted.return_value = False
        mock_jpype.getDefaultJVMPath.return_value = "fake_jvm"

        mock_sc = MagicMock()
        mock_sc.isModelInitialized.return_value = True
        mock_jpype.JClass.return_value = mock_sc

        with patch("autoplot_bridge.connection.Path") as mock_path:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = mock_path_inst

            conn.init_autoplot(headless=True)

        # Verify -Djava.awt.headless=true was passed
        start_call = mock_jpype.startJVM.call_args
        args = start_call[0]
        assert "-Djava.awt.headless=true" in args
        assert conn.is_headless() is True

    @patch("autoplot_bridge.connection.jpype")
    @patch("autoplot_bridge.connection.AUTOPLOT_JAR", "fake.jar")
    @patch("autoplot_bridge.connection.JAVA_HOME", None)
    def test_headless_false_omits_flag(self, mock_jpype):
        """headless=False should NOT pass the headless flag."""
        import autoplot_bridge.connection as conn
        conn._headless_mode = True

        mock_jpype.isJVMStarted.return_value = False
        mock_jpype.getDefaultJVMPath.return_value = "fake_jvm"

        mock_sc = MagicMock()
        mock_sc.isModelInitialized.return_value = True
        mock_jpype.JClass.return_value = mock_sc

        with patch("autoplot_bridge.connection.Path") as mock_path:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = mock_path_inst

            conn.init_autoplot(headless=False)

        start_call = mock_jpype.startJVM.call_args
        args = start_call[0]
        assert "-Djava.awt.headless=true" not in args
        assert conn.is_headless() is False


class TestAutoplotPromptGUIMode:
    """Test that the autoplot agent prompt includes GUI-mode instructions."""

    def test_default_no_gui_section(self):
        from knowledge.prompt_builder import build_autoplot_prompt
        prompt = build_autoplot_prompt(gui_mode=False)
        assert "Interactive GUI Mode" not in prompt

    def test_gui_mode_appends_section(self):
        from knowledge.prompt_builder import build_autoplot_prompt
        prompt = build_autoplot_prompt(gui_mode=True)
        assert "Interactive GUI Mode" in prompt
        assert "reset" in prompt
        assert "save_session" in prompt
        assert "load_session" in prompt

    def test_autoplot_prompt_has_method_catalog(self):
        from knowledge.prompt_builder import build_autoplot_prompt
        prompt = build_autoplot_prompt(gui_mode=False)
        assert "Available Methods" in prompt
        assert "plot_cdaweb" in prompt


class TestToolSchemaCount:
    """Test that the correct number of tool schemas are registered."""

    def test_total_tool_count_is_14(self):
        from agent.tools import get_tool_schemas
        schemas = get_tool_schemas()
        assert len(schemas) == 14

    def test_autoplot_tools_present(self):
        from agent.tools import get_tool_schemas
        names = {t["name"] for t in get_tool_schemas()}
        assert "execute_autoplot" in names
        assert "delegate_to_autoplot" in names

    def test_execute_autoplot_is_autoplot_category(self):
        from agent.tools import get_tool_schemas
        tool = next(t for t in get_tool_schemas() if t["name"] == "execute_autoplot")
        assert tool["category"] == "autoplot"


class TestAutoplotCommandsReset:
    """Test that reset() clears internal state."""

    def test_reset_clears_state(self):
        from autoplot_bridge.commands import AutoplotCommands

        cmd = AutoplotCommands.__new__(AutoplotCommands)
        cmd.verbose = False
        cmd.gui_mode = False
        cmd._current_uri = "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude"
        cmd._current_time_range = "2024-01-01 to 2024-01-02"
        cmd._label_colors = {"Bmag": "red"}

        # Mock ctx to avoid JVM
        mock_ctx = MagicMock()
        cmd._ctx = mock_ctx

        cmd.reset()

        assert cmd._current_uri is None
        assert cmd._current_time_range is None
        assert cmd._label_colors == {}
        mock_ctx.reset.assert_called_once()
        mock_ctx.waitUntilIdle.assert_called_once()


class TestSingletonGuard:
    """Test that get_commands raises on conflicting gui_mode."""

    def test_conflicting_gui_mode_raises(self):
        import autoplot_bridge.commands as mod

        # Reset singleton
        mod._commands = None

        cmd = mod.get_commands(gui_mode=False)
        assert cmd.gui_mode is False

        with pytest.raises(RuntimeError, match="cannot change to gui_mode=True"):
            mod.get_commands(gui_mode=True)

        # Clean up
        mod._commands = None

    def test_same_gui_mode_reuses_instance(self):
        import autoplot_bridge.commands as mod
        mod._commands = None

        cmd1 = mod.get_commands(gui_mode=False)
        cmd2 = mod.get_commands(gui_mode=False)
        assert cmd1 is cmd2

        mod._commands = None


class TestDOMMutationsCallWaitUntilIdle:
    """Test that DOM-mutating methods call waitUntilIdle() after changes."""

    def _make_commands(self):
        from autoplot_bridge.commands import AutoplotCommands
        cmd = AutoplotCommands.__new__(AutoplotCommands)
        cmd.verbose = False
        cmd.gui_mode = True
        cmd._current_uri = None
        cmd._current_time_range = None
        cmd._label_colors = {}

        mock_ctx = MagicMock()
        cmd._ctx = mock_ctx
        return cmd, mock_ctx

    def test_set_plot_title_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        cmd.set_plot_title("Test Title")
        mock_ctx.waitUntilIdle.assert_called()

    def test_set_axis_label_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        cmd.set_axis_label("y", "B (nT)")
        mock_ctx.waitUntilIdle.assert_called()

    def test_toggle_log_scale_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        cmd.toggle_log_scale("y", True)
        mock_ctx.waitUntilIdle.assert_called()

    def test_set_axis_range_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        with patch("autoplot_bridge.commands.jpype") as mock_jpype:
            mock_jpype.JClass.return_value = MagicMock()
            cmd.set_axis_range("y", 0, 100)
        mock_ctx.waitUntilIdle.assert_called()

    def test_load_session_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        cmd.load_session("test.vap")
        mock_ctx.waitUntilIdle.assert_called()

    def test_reset_calls_wait(self):
        cmd, mock_ctx = self._make_commands()
        cmd.reset()
        mock_ctx.waitUntilIdle.assert_called()


class TestExportPNGGUIMode:
    """Test that export_png skips auto-open in GUI mode via dispatch."""

    def test_gui_mode_skips_auto_open(self):
        """In GUI mode, export_png should not auto-open the file."""
        from agent.core import OrchestratorAgent

        # Create agent with gui_mode=True, mocking out Gemini
        with patch("agent.core.genai"), \
             patch("agent.core.get_system_prompt", return_value="test"), \
             patch("agent.core.get_tool_schemas", return_value=[]):
            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            agent.verbose = False
            agent.gui_mode = True
            agent.logger = MagicMock()

            # Mock autoplot
            mock_autoplot = MagicMock()
            mock_autoplot.export_png.return_value = {
                "status": "success",
                "filepath": "/tmp/test.png",
                "size_bytes": 1000,
            }
            agent._autoplot = mock_autoplot

            result = agent._dispatch_autoplot_method("export_png", {"filename": "test.png"})

        assert result["status"] == "success"
        # auto_opened should not be set because gui_mode skips it
        assert "auto_opened" not in result
