<#
  status.ps1 - what is running: master and worker processes, listening ports, and the master's own view (registered workers, cores, memory, applications).
  usage: powershell -ExecutionPolicy Bypass -File scripts\spark\status.ps1
#>
. "$PSScriptRoot\env.ps1"
$pids = Get-Pids
Write-Host ("Machine IPv4 used for Spark: {0}   Java: {1}" -f $LanIp, $JavaHome)
if (-not $pids.master) { Write-Host "Master: not started by this project." } else {
    $m = $pids.master
    Write-Host ("Master: pid {0} alive={1} port {2} open={3}  UI http://{4}:{5}" -f $m.pid, (Test-Alive $m.pid), $m.port, (Test-PortOpen $m.host $m.port), $m.host, $m.webui)
}
foreach ($w in @($pids.workers.PSObject.Properties)) {
    $v = $w.Value
    Write-Host ("Worker '{0}': pid {1} alive={2} port {3} open={4} cores {5} memory {6}" -f $w.Name, $v.pid, (Test-Alive $v.pid), $v.port, (Test-PortOpen $v.host $v.port), $v.cores, $v.memory)
}
if ($pids.master) {
    try {
        $j = Invoke-RestMethod -Uri ("http://{0}:{1}/json" -f $pids.master.host, $pids.master.webui) -TimeoutSec 5
        Write-Host ("`nMaster view: status {0}, {1} worker(s), cores used {2}/{3}, memory used {4}/{5} MB" -f $j.status, @($j.workers).Count, $j.coresused, $j.cores, $j.memoryused, $j.memory)
        foreach ($w in @($j.workers)) { Write-Host ("  {0}  {1}  cores {2} (used {3})  memory {4} MB (used {5})  state {6}" -f $w.id, $w.host, $w.cores, $w.coresused, $w.memory, $w.memoryused, $w.state) }
        foreach ($a in @($j.activeapps)) { Write-Host ("  app {0} '{1}' cores {2} state {3}" -f $a.id, $a.name, $a.cores, $a.state) }
    } catch { Write-Host "Master REST not reachable: $($_.Exception.Message)" -ForegroundColor Yellow }
}
$java = @(Get-Process java -ErrorAction SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "pyspark\.(worker|daemon)" }).Count
Write-Host ("`nProcesses now: java.exe {0}, pyspark python workers {1}" -f $java, $py)
