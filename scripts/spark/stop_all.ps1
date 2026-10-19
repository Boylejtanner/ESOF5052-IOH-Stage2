<#
  stop_all.ps1 - stop every Spark process THIS project started (master, workers, their executors and python workers). Idempotent; changes nothing else.
  It uses the pids recorded by start_master.ps1 / start_worker.ps1 and additionally removes java.exe and python.exe processes whose command line points into this project's
  pyspark folder (leftovers of a crashed session). Other java or python programs are never touched.
#>
. "$PSScriptRoot\env.ps1"
$pids = Get-Pids
$stopped = 0
foreach ($w in @($pids.workers.PSObject.Properties)) {
    if (Test-Alive $w.Value.pid) { Write-Host "Stopping worker '$($w.Name)' (pid $($w.Value.pid))"; Stop-Tree $w.Value.pid; $stopped++ }
}
if ($pids.master -and (Test-Alive $pids.master.pid)) { Write-Host "Stopping master (pid $($pids.master.pid))"; Stop-Tree $pids.master.pid; $stopped++ }

$sparkMarker = $SparkHome.ToLower()
$venvMarker = (Split-Path $PythonExe).ToLower()
$left = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and (
        (($_.Name -eq "java.exe") -and $_.CommandLine.ToLower().Contains($sparkMarker)) -or
        (($_.Name -eq "python.exe") -and ($_.CommandLine -match "pyspark\.(worker|daemon)") -and $_.CommandLine.ToLower().Contains($venvMarker.Replace("\scripts", "")))
    ) })
foreach ($p in $left) { Write-Host "Stopping leftover $($p.Name) (pid $($p.ProcessId))"; Stop-Tree $p.ProcessId; $stopped++ }
Save-Pids ([pscustomobject]@{ master = $null; workers = [pscustomobject]@{} })
Write-Host "Done: $stopped process tree(s) stopped."
