from pathlib import Path
from typing import Optional


class PromptManager:
    def __init__(self, system_path: str, tools_path: Optional[str] = None):
        self.system_path = Path(system_path)
        self.tools_path = Path(tools_path) if tools_path else None


    def system_prompt(self) -> str:
        system = self.system_path.read_text(encoding="utf-8")
        tools = self.tools_path.read_text(encoding="utf-8") if self.tools_path and self.tools_path.exists() else ""
        return system.replace("{{TOOLS_SECTION}}", tools)