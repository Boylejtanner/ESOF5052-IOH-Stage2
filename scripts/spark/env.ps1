<#
  env.ps1 - Spark / Java / Python environment for THIS PowerShell process only. Dot-source it:   . scripts\spark\env.ps1
  Nothing is written to the registry or to the machine/user environment.

  Variables it defines for the calling script: $BshiRoot, $Cluster (the "cluster" block of configs\hemodynamics.json), $PythonExe, $JavaHome, $SparkHome, $LanIp, $LogDir, $PidFile.
  Override points (environment variables, process-level): BSHI_PYTHON, BSHI_JAVA_HOME, BSHI_BIND_HOST (the IPv4 this machine uses for Spark), BSHI_MASTER_HOST (where the master runs).
#>
$ErrorActionPreference = "Stop"
$BshiRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ConfigFile = Join-Path $BshiRoot "configs\hemodynamics.json"
if (Test-Path $ConfigFile) { $Cluster = (Get-Content $ConfigFile -Raw | ConvertFrom-Json).cluster } else { $Cluster = $null }
if (-not $Cluster) { throw "No 'cluster' block in $ConfigFile" }
$LogDir = Join-Path $BshiRoot "logs\spark"
$PidFile = Join-Path $LogDir "pids.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-VpnIPv4 {
    # IPv4 of a private-network adapter (Tailscale uses 100.64.0.0/10; ZeroTier / WireGuard are found by adapter name), or $null.
    $c = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object {
        $_.IPAddress -match '^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.' -or $_.InterfaceAlias -match 'Tailscale|ZeroTier|WireGuard' } | Select-Object -First 1
    if ($c) { return $c.IPAddress } else { return $null }
}
function Get-LanIPv4 {
    # IPv4 of the default-route adapter, never a Hyper-V / WSL / virtual adapter (the Hyper-V "Default Switch" is 192.168.64.x) and never link-local.
    # Network 'vpn' (env BSHI_NETWORK or cluster.master_host = "vpn") uses the private-network address instead, so a laptop elsewhere can join; without a VPN adapter it falls back to the LAN
    # address (the desktop alone keeps working).
    if ($env:BSHI_BIND_HOST) { return $env:BSHI_BIND_HOST }
    $net = if ($env:BSHI_NETWORK) { $env:BSHI_NETWORK } elseif ($Cluster -and $Cluster.master_host -eq "vpn") { "vpn" } else { "lan" }
    if ($net -eq "vpn") {
        $v = Get-VpnIPv4
        if ($v) { return $v }
        Write-Warning "Network 'vpn' requested but no Tailscale/ZeroTier/WireGuard address was found: using the LAN address (desktop-only keeps working)."
    }
    $routes = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | ForEach-Object {
        $m = (Get-NetIPInterface -InterfaceIndex $_.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).InterfaceMetric
        [pscustomobject]@{ Index = $_.InterfaceIndex; Metric = $_.RouteMetric + $m }
    } | Sort-Object Metric
    foreach ($rt in $routes) {
        $nic = Get-NetAdapter -InterfaceIndex $rt.Index -ErrorAction SilentlyContinue
        if (-not $nic -or $nic.Status -ne "Up") { continue }
        if ($nic.Name -match "vEthernet|Hyper-V|WSL|Loopback|VirtualBox|VMware|Bluetooth") { continue }
        $ip = (Get-NetIPAddress -InterfaceIndex $rt.Index -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
        if ($ip -and $ip -notlike "192.168.64.*") { return $ip }
    }
    throw "No usable LAN IPv4 (no default-route adapter that is up). Set `$env:BSHI_BIND_HOST to the Wi-Fi or Ethernet IPv4 and retry."
}

function Find-Jdk {
    $candidates = @($env:BSHI_JAVA_HOME, $Cluster.java_home_desktop)
    foreach ($c in $candidates) { if ($c -and (Test-Path (Join-Path $c "bin\java.exe"))) { return $c } }
    $patterns = @("C:\Program Files\Eclipse Adoptium\jdk-17*", "C:\Program Files\Eclipse Adoptium\jdk-21*", "C:\Program Files\Java\jdk-17*", "C:\Program Files\Java\jdk-21*",
                  "C:\Program Files\Microsoft\jdk-17*", "C:\Program Files\Microsoft\jdk-21*", "C:\Program Files\Amazon Corretto\jdk17*", "C:\Program Files\Zulu\zulu-17*")
    foreach ($p in $patterns) {
        $m = Get-ChildItem $p -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($m) { return $m.FullName }
    }
    throw "Java 17 or 21 not found. Spark 4 needs it (Temurin 17: https://adoptium.net/temurin/releases/?version=17). Or set `$env:BSHI_JAVA_HOME."
}

function Get-Pids { if (Test-Path $PidFile) { return (Get-Content $PidFile -Raw | ConvertFrom-Json) } else { return [pscustomobject]@{ master = $null; workers = [pscustomobject]@{} } } }
function Save-Pids($obj) { $obj | ConvertTo-Json -Depth 6 | Set-Content -Path $PidFile -Encoding UTF8 }
function Test-Alive($processId) { if (-not $processId) { return $false }; return [bool](Get-Process -Id $processId -ErrorAction SilentlyContinue) }
function Test-PortOpen($ip, $port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $a = $c.BeginConnect($ip, [int]$port, $null, $null); $ok = $a.AsyncWaitHandle.WaitOne(1000, $false) -and $c.Connected; return $ok } catch { return $false } finally { $c.Close() }
}
function Stop-Tree($processId) {
    # children first, then the process itself (cmd.exe -> java.exe -> executor java / python workers)
    foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue)) { Stop-Tree $child.ProcessId }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

$PythonExe = if ($env:BSHI_PYTHON) { $env:BSHI_PYTHON } else { Join-Path $BshiRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $PythonExe)) { throw "Python not found: $PythonExe (set `$env:BSHI_PYTHON)." }
$JavaHome = Find-Jdk
$SparkHome = (& $PythonExe -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))").Trim()
if (-not (Test-Path (Join-Path $SparkHome "bin\spark-class.cmd"))) { throw "pyspark has no bin\spark-class.cmd under $SparkHome" }
$LanIp = Get-LanIPv4

# process-level only
$env:JAVA_HOME = $JavaHome
$env:SPARK_HOME = $SparkHome
$env:PYSPARK_PYTHON = "python"                      # resolved on each executor through ITS PATH (the laptop's venv); a full desktop path would not exist on the laptop
$env:PYSPARK_DRIVER_PYTHON = $PythonExe
$env:SPARK_LOCAL_IP = $LanIp
$env:BSHI_ROOT = $BshiRoot
$env:PYTHONPATH = (Join-Path $BshiRoot "scripts\spark\pysite") + ";" + (Join-Path $BshiRoot "notebooks")   # executors import hemodynamics from THIS machine's copy (Spark cannot ship files on Windows without winutils); pysite flushes the worker socket (PySpark 4.0.1 Windows bug)
$env:PYTHONDONTWRITEBYTECODE = "1"
# Hadoop's Shell class refuses to load on Windows unless HADOOP_HOME is an EXISTING folder (this PC has a stale C:\Hadoop variable but no such folder). An empty project-local folder
# satisfies it for this process only. No winutils is downloaded: Spark must never WRITE local files (read with Spark, write from Python).
if (-not $env:HADOOP_HOME -or -not (Test-Path $env:HADOOP_HOME)) {
    $HadoopHome = Join-Path $LogDir "hadoop_home"
    New-Item -ItemType Directory -Force -Path (Join-Path $HadoopHome "bin") | Out-Null
    $env:HADOOP_HOME = $HadoopHome
}
# executors resolve "python" through the worker's own PATH, so the laptop uses ITS venv, not the desktop's path
$env:PATH = (Split-Path $PythonExe) + ";" + (Join-Path $JavaHome "bin") + ";" + $env:PATH
