# SSL-corrected Windows downloader

Your previous Python downloader failed certificate validation. This version uses Windows `curl.exe` (Windows native certificate handling) while keeping SSL verification enabled. It does NOT use `--insecure` and does NOT disable certificate checks.

1. Extract this ZIP.
2. Open PowerShell in the extracted folder.
3. First test that Windows curl exists:

```powershell
curl.exe --version
```

4. Run:

```powershell
python download_all_500_pdfs_windows.py
```

The script attempts all 500 source URLs and creates `uoh_all500_source_pdfs_and_log.zip`.

If curl also reports a certificate error, do not bypass it. That indicates the website's certificate chain or a local HTTPS inspection/proxy problem. Keep the log and share it for a safe next step.
