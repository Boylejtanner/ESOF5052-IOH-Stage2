<#
  start_master.ps1 - start the Spark standalone master on THIS machine (Windows native, Java 17). Idempotent: does nothing if it is already running.
  usage: powershell -ExecutionPolicy Bypass -File scripts\spark\start_master.ps1 [-BindHost 10.0.0.19] [-Network vpn] [-Port 7077] [-WebUi 8080]
  The master binds to the Wi-Fi/Ethernet IPv4 (never the Hyper-V 192.168.64.x adapter). With the direct cable use -BindHost 192.168.50.1. With -Network vpn it binds to the Tailscale /
  WireGuard address, which is how a laptop on another network joins (see docs\README_LAPTOP.md, part B); the desktop worker and the driver then use that address too.
#>
param([string]$BindHost, [int]$Port = 0, [int]$WebUi = 0, [string]$Network = "")
if ($Network) { $env:BSHI_NETWORK = $Network }
. "$PSScriptRoot\env.ps1"
if ($BindHost) { $LanIp = $BindHost; $env:SPARK_LOCAL_IP = $BindHost }
if ($LanIp -like "192.168.64.*") { throw "Refusing to bind to the Hyper-V Default Switch ($LanIp)." }
if ($Port -eq 0) { $Port = [int]$Cluster.master_port }
if ($WebUi -eq 0) { $WebUi = [int]$Cluster.master_ui }

$pids = Get-Pids
if ($pids.master -and (Test-Alive $pids.master.pid) -and (Test-PortOpen $pids.master.host $pids.master.port)) {
    Write-Host "Master already running: spark://$($pids.master.host):$($pids.master.port) (pid $($pids.master.pid)); UI http://$($pids.master.host):$($pids.master.webui)"
    return
}
if (Test-PortOpen $LanIp $Port) { throw "Port $Port on $LanIp is already in use by another program. Stop it or choose -Port." }

$sparkClass = Join-Path $SparkHome "bin\spark-class.cmd"
$out = Join-Path $LogDir "master.out.log"
$err = Join-Path $LogDir "master.err.log"
$cmdLine = '/c ""' + $sparkClass + '" org.apache.spark.deploy.master.Master --host ' + $LanIp + ' --port ' + $Port + ' --webui-port ' + $WebUi + '"'
$p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Write-Host "Starting master (pid $($p.Id)) on $LanIp`:$Port ..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) { break }
    if (Test-PortOpen $LanIp $Port) { $ok = $true; break }
}
if (-not $ok) {
    Write-Host "Master did not come up. Last log lines:" -ForegroundColor Red
    Get-Content $err -Tail 15 -ErrorAction SilentlyContinue
    Get-Content $out -Tail 15 -ErrorAction SilentlyContinue
    Stop-Tree $p.Id
    throw "Master failed to start."
}
$pids.master = [pscustomobject]@{ pid = $p.Id; host = $LanIp; port = $Port; webui = $WebUi; started = (Get-Date).ToString("s") }
Save-Pids $pids
Write-Host "Master up: spark://$LanIp`:$Port   UI http://$LanIp`:$WebUi   logs $LogDir"
