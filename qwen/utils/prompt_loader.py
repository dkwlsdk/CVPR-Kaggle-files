from pathlib import Path


class PromptLoader:
    """Loads and renders external prompt files."""

    def __init__(self, system_prompt_file: Path, user_prompt_file: Path):
        self.system_prompt_file = system_prompt_file
        self.user_prompt_file = user_prompt_file
        self._system_prompt = None
        self._user_template = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self.system_prompt_file.read_text(
                encoding="utf-8"
            ).strip()
        return self._system_prompt

    @property
    def user_template(self) -> str:
        if self._user_template is None:
            self._user_template = self.user_prompt_file.read_text(
                encoding="utf-8"
            ).strip()
        return self._user_template

    def render_user_prompt(self, *, duration: float, fps: float, n_frames: int) -> str:
        return self.user_template.format(duration=duration, fps=fps, n_frames=n_frames)

    def reload(self) -> None:
        self._system_prompt = None
        self._user_template = None
