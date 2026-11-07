param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [string]$Python = '.venv/Scripts/python.exe',
    [string]$Config = 'configs/deepseek.local.json',
    [int]$Port = 8766,
    [string[]]$AllowRoot = @()
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Install the environment first.' }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw 'Copy and review the DeepSeek config template first.' }
$configPath = (Resolve-Path -LiteralPath $Config).Path
$settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($settings.input_usd_per_million -le 0 -or $settings.output_usd_per_million -le 0 -or $settings.pricing_reference -like 'REPLACE*') {
    throw 'Check official current prices, record the reference/date and review the request/cost limits first.'
}
$credential = Read-Host 'DeepSeek API Key (hidden, used only by the application process)' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($credential)
$process = $null
try {
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = (Resolve-Path -LiteralPath $Python).Path
    $info.UseShellExecute = $false
    if (-not $info.PSObject.Properties['ArgumentList']) { throw 'Use PowerShell 7 or newer for safe argument handling.' }
    foreach ($argument in @('-m', 'experienceguard', 'serve', '--data-dir', $DataDir, '--port', "$Port", '--enable-remote')) {
        $info.ArgumentList.Add($argument)
    }
    foreach ($root in $AllowRoot) { $info.ArgumentList.Add('--allow-root'); $info.ArgumentList.Add($root) }
    $info.Environment['EG_V2_API_KEY'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $info.Environment['EG_V2_REMOTE_ENABLE'] = 'true'
    $info.Environment['EG_DEEPSEEK_CONFIG'] = $configPath
    $process = [System.Diagnostics.Process]::Start($info)
    $info.Environment.Remove('EG_V2_API_KEY') | Out-Null
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "Application exited with code $($process.ExitCode)." }
}
finally {
    if ($null -ne $process -and -not $process.HasExited) { $process.Kill() }
    if ($null -ne $process) { $process.Dispose() }
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $credential.Dispose()
}
