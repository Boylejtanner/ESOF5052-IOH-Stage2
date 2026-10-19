<#
  start_worker.ps1 - start one Spark standalone worker on THIS machine and register it with the master. Idempotent per -Name.
  usage (desktop):  start_worker.ps1 -Name desktop -Port 7078 -WebUi 8081 -Cores 8 -Memory 24g -WorkDir logs\spark\work\desktop
  usage (laptop):   start_worker.ps1 -Name laptop -Master spark://192.168.50.1:7077 -Cores 6 -Memory 8g
  usage (simulated second worker on the desktop): start_worker.ps1 -Name sim2 -Port 7079 -WebUi 8082 -Cores 4 -Memory 6g -WorkDir logs\spark\work\sim2
  The worker name is exported to its executors as BSHI_WORKER_NAME (shown by cluster_report); executors use this machine's own python from PATH and read data
  from this machine's own project root (BSHI_ROOT), so the laptop needs its own copy of the data shard, not the desktop's path.
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Port = 0, [int]$WebUi = 8081, [int]$Cores = 0, [string]$Memory = "",
    [string]$WorkDir = "", [string]$Master = "", [string]$BindHost = "", [string]$Network = ""
)
if ($Network) { $env:BSHI_NETWORK = $Network }
. "$PSScriptRoot\env.ps1"
if ($BindHost) { $LanIp = $BindHost; $env:SPARK_LOCAL_IP = $BindHost }
if ($LanIp -like "192.168.64.*") { throw "Refusing to bind to the Hyper-V Default Switch ($LanIp)." }
if ($Port -eq 0) { $Port = [int]$Cluster.worker_port }
if ($Cores -eq 0) { $Cores = [int]$Cluster.desktop_worker_cores }
if (-not $Memory) { $Memory = [string]$Cluster.desktop_worker_memory }
if (-not $WorkDir) { $WorkDir = Join-Path $LogDir ("work\" + $Name) } elseif (-not [System.IO.Path]::IsPathRooted($WorkDir)) { $WorkDir = Join-Path $BshiRoot $WorkDir }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

$pids = Get-Pids
if (-not $Master) {
    if ($env:BSHI_MASTER_HOST) { $Master = "spark://$($env:BSHI_MASTER_HOST):$($Cluster.master_port)" }
    elseif ($pids.master) { $Master = "spark://$($pids.master.host):$($pids.master.port)" }
    else { throw "No master known. Start it first (start_master.ps1) or pass -Master spark://<host>:7077." }
}
$existing = $pids.workers.PSObject.Properties[$Name]
if ($existing -and (Test-Alive $existing.Value.pid) -and (Test-PortOpen $LanIp $existing.Value.port)) {
    Write-Host "Worker '$Name' already running (pid $($existing.Value.pid), port $($existing.Value.port))."
    return
}
if (Test-PortOpen $LanIp $Port) { throw "Port $Port on $LanIp is already in use (another worker?). Choose a different -Port for '$Name'." }

$env:BSHI_WORKER_NAME = $Name
$sparkClass = Join-Path $SparkHome "bin\spark-class.cmd"
$out = Join-Path $LogDir ("worker-" + $Name + ".out.log")
$err = Join-Path $LogDir ("worker-" + $Name + ".err.log")
$cmdLine = '/c ""' + $sparkClass + '" org.apache.spark.deploy.worker.Worker ' + $Master + ' --host ' + $LanIp + ' --cores ' + $Cores + ' --memory ' + $Memory +
           ' --port ' + $Port + ' --webui-port ' + $WebUi + ' --work-dir "' + $WorkDir + '""'
$p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Write-Host "Starting worker '$Name' (pid $($p.Id)): $Cores cores, $Memory, port $Port, UI $WebUi -> $Master"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) { break }
    if (Test-PortOpen $LanIp $Port) { $ok = $true; break }
}
if (-not $ok) {
    Write-Host "Worker did not come up. Last log lines:" -ForegroundColor Red
    Get-Content $err -Tail 15 -ErrorAction SilentlyContinue
    Get-Content $out -Tail 15 -ErrorAction SilentlyContinue
    Stop-Tree $p.Id
    throw "Worker '$Name' failed to start."
}
$entry = [pscustomobject]@{ pid = $p.Id; host = $LanIp; port = $Port; webui = $WebUi; cores = $Cores; memory = $Memory; master = $Master; work_dir = $WorkDir; started = (Get-Date).ToString("s") }
$pids.workers | Add-Member -NotePropertyName $Name -NotePropertyValue $entry -Force
Save-Pids $pids
Write-Host "Worker '$Name' listening on $LanIp`:$Port (it registers with the master within a few seconds; see status.ps1)."
