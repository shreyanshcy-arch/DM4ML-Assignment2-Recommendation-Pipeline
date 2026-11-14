$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PipelineScript = Join-Path $ProjectRoot "run_pipeline.py"
$LogDirectory = Join-Path $ProjectRoot "reports\scheduled_runs"

New-Item -ItemType Directory -Force $LogDirectory | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDirectory "pipeline_$Timestamp.log"
$StdOutFile = Join-Path $LogDirectory "stdout_$Timestamp.tmp"
$StdErrFile = Join-Path $LogDirectory "stderr_$Timestamp.tmp"

Set-Location $ProjectRoot

"Pipeline started: $(Get-Date)" | Out-File $LogFile -Encoding utf8

try {
    if (!(Test-Path $PythonExe)) {
        throw "Virtual environment Python not found: $PythonExe"
    }

    if (!(Test-Path $PipelineScript)) {
        throw "Pipeline script not found: $PipelineScript"
    }

    $Process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList "`"$PipelineScript`"" `
        -WorkingDirectory $ProjectRoot `
        -Wait `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $StdOutFile `
        -RedirectStandardError $StdErrFile

    if (Test-Path $StdOutFile) {
        "========== STANDARD OUTPUT ==========" |
            Out-File $LogFile -Append -Encoding utf8
        Get-Content $StdOutFile |
            Out-File $LogFile -Append -Encoding utf8
    }

    if (Test-Path $StdErrFile) {
        "========== STANDARD ERROR / WARNINGS ==========" |
            Out-File $LogFile -Append -Encoding utf8
        Get-Content $StdErrFile |
            Out-File $LogFile -Append -Encoding utf8
    }

    if ($Process.ExitCode -ne 0) {
        throw "Python pipeline failed with exit code $($Process.ExitCode)"
    }

    "STATUS: SUCCESS" | Out-File $LogFile -Append -Encoding utf8
    "Exit code: $($Process.ExitCode)" |
        Out-File $LogFile -Append -Encoding utf8
    "Pipeline completed: $(Get-Date)" |
        Out-File $LogFile -Append -Encoding utf8
}
catch {
    "STATUS: FAILED" | Out-File $LogFile -Append -Encoding utf8
    $_ | Out-String | Out-File $LogFile -Append -Encoding utf8
    exit 1
}
finally {
    Remove-Item $StdOutFile, $StdErrFile -Force -ErrorAction SilentlyContinue
}

exit 0
