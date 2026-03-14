from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

def WebSearch():
    search_tool = DuckDuckGoSearchRun()
    shell_results = search_tool.invoke("current war news")
    print(shell_results)
    """
    20 minutes ago ·LIVE Iran IsraelWarLive Updates: Iran Says 'No Plan to Close Strait of Hormuz' 
    But Defends Right to 'Preserve Security' Iran IsraelWarLive Updates: Iran's new Supreme Leader 
    Mojtaba Khamenei, the son of assassinated Ayatollah Ali Khamenei, in his first ever statement 
    warned the United States that it will revenge the death of its martyrs while threatening 
    America to close its bases in ...
    """

def ShellCommandTool():
    shell_tool = ShellTool()
    shell_results = shell_tool.invoke("whoami")
    print(shell_results)
    """
    UserWarning: The shell tool has no safeguards by default. Use at your own risk.
    warnings.warn(
    Executing command:
    whoami
    azuread\vinayakkanchan
    """



if __name__ == "__main__":
    ShellCommandTool()
