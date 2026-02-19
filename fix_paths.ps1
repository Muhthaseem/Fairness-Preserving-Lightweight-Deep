# fix_paths.ps1
# Replaces old paths from source machines with current Windows paths,
# and normalizes all forward slashes to backslashes in path fields.
#
# Old paths found:
#   /media/admin-account/HDD_Studies/DeepFake  (Linux machine)
#   f:/DeepFake                                 (another Windows machine)
#   D:\M3 Projects\DeepFake_Research/...        (already partially fixed, mixed slashes)
#
# Target: D:\M3 Projects\DeepFake_Research\...  (all backslashes)

$newBase = "D:\M3 Projects\DeepFake_Research"

# All old prefixes to replace (order matters: most specific first)
$replacements = @(
    @{ Old = "/media/admin-account/HDD_Studies/DeepFake"; New = $newBase },
    @{ Old = "f:/DeepFake"; New = $newBase },
    @{ Old = "f:\DeepFake"; New = $newBase }
)

$csvFiles = Get-ChildItem -Path "D:\M3 Projects\DeepFake_Research\outputs" -Filter "*.csv" -Recurse

Write-Host "Found $($csvFiles.Count) CSV files to process..."
Write-Host ""

foreach ($file in $csvFiles) {
    $sizeMB = [math]::Round($file.Length / 1MB, 1)
    Write-Host "Processing: $($file.Name) ($sizeMB MB)..."

    $startTime = Get-Date
    $tempFile = $file.FullName + ".tmp"

    $reader = [System.IO.StreamReader]::new($file.FullName, [System.Text.Encoding]::UTF8)
    $writer = [System.IO.StreamWriter]::new($tempFile, $false, [System.Text.Encoding]::UTF8)

    $lineCount = 0
    $isHeader = $true

    while ($null -ne ($line = $reader.ReadLine())) {
        if ($isHeader) {
            # Write header unchanged
            $writer.WriteLine($line)
            $isHeader = $false
            $lineCount++
            continue
        }

        # Apply all old-path replacements
        foreach ($r in $replacements) {
            if ($line.Contains($r.Old)) {
                $line = $line.Replace($r.Old, $r.New)
            }
        }

        # Normalize: replace forward slashes with backslashes ONLY inside path fields.
        # Path fields are the first three columns (face_path, frame_path, video_path).
        # We split on comma, fix slashes in the first 3 fields, then rejoin.
        # Use a simple approach: replace / with \ only in segments that start with a drive letter.
        # Safer: just replace all forward slashes that appear after "D:\M3 Projects\DeepFake_Research"
        $line = $line.Replace("$newBase/", "$newBase\")
        # Also catch any remaining forward slashes within the new base path segments
        # by doing a targeted replacement on the known sub-paths
        $line = $line -replace [regex]::Escape("$newBase\outputs/"), "$newBase\outputs\"
        $line = $line -replace [regex]::Escape("$newBase\Datasets/"), "$newBase\Datasets\"
        # General cleanup: replace any remaining / after the base
        # Split CSV carefully: only touch first 3 columns
        $cols = $line.Split(',')
        for ($i = 0; $i -lt [Math]::Min(3, $cols.Count); $i++) {
            if ($cols[$i].StartsWith($newBase)) {
                $cols[$i] = $cols[$i].Replace('/', '\')
            }
        }
        $line = $cols -join ','

        $writer.WriteLine($line)
        $lineCount++
    }

    $reader.Close()
    $writer.Close()

    Move-Item -Path $tempFile -Destination $file.FullName -Force

    $elapsed = (Get-Date) - $startTime
    Write-Host "  Done: $lineCount lines in $([math]::Round($elapsed.TotalSeconds,1))s"
}

Write-Host ""
Write-Host "All CSV files updated. New base path: $newBase"
