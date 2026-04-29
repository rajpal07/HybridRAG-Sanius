from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    out_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _pick_anchor_href(html: str, *, base_url: str, contains_text: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    needle = contains_text.strip().lower()
    for a in soup.find_all("a"):
        text = (a.get_text(" ", strip=True) or "").strip().lower()
        if needle in text:
            href = a.get("href")
            if href:
                return urljoin(base_url, href)
    raise RuntimeError(f"Could not find link containing text: {contains_text!r} on {base_url}")


def _download_file(client: httpx.Client, url: str, out_path: Path) -> None:
    _ensure_parent(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"skip (exists): {out_path}")
        return

    def _stream(extra_headers: dict[str, str] | None = None):
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        return client.stream("GET", url, headers=headers, follow_redirects=True, timeout=300.0)

    with _stream() as r:
        if r.status_code == 403:
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}/"
            r.close()
            with _stream({"Referer": origin}) as r2:
                r2.raise_for_status()
                r = r2

        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with tmp_path.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_bytes(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
        tmp_path.replace(out_path)
        print(f"saved: {out_path}")


def main() -> int:
    root = _repo_root()
    raw = root / "data" / "raw"

    # Direct URLs (these generally respond with the file directly)
    direct_specs: list[DownloadSpec] = [
        DownloadSpec(
            name="NICE CG143 full guideline PDF",
            url="https://www.nice.org.uk/guidance/cg143/evidence/full-guideline-pdf-186634333",
            out_path=raw / "pdfs" / "nice_cg143_full_guideline.pdf",
        ),
        DownloadSpec(
            name="NICE QS58 Sickle cell disease PDF",
            url="https://www.nice.org.uk/guidance/qs58/resources/sickle-cell-disease-pdf-2098733894341",
            out_path=raw / "pdfs" / "nice_qs58_sickle_cell_disease.pdf",
        ),
        DownloadSpec(
            name="West London HCC adult sickle cell guideline PDF",
            url="https://www.westlondonhcc.nhs.uk/-/media/hcc/documents/hcc-adult-sickle-cell-guideline-version-11-sept2024-1.pdf?hash=2EC4AF3DB4EA6855C5196F47CDC36596&rev=371b446741954e6f903788801bde31f8",
            out_path=raw / "pdfs" / "west_london_hcc_adult_sickle_cell_guideline_v11_sep2024.pdf",
        ),
        DownloadSpec(
            name="NHS HES provider-level analysis CSV (2024-25)",
            url="https://files.digital.nhs.uk/3C/145B13/hosp-epis-stat-admi-pla-2024-25.csv",
            out_path=raw / "hes" / "hosp-epis-stat-admi-pla-2024-25.csv",
        ),
    ]

    # Indirect URLs (need to resolve the real download link from a page)
    synthea_page = "https://synthetichealth.github.io/synthea/"
    mimic_demo_zip = (
        "https://physionet.org/static/published-projects/mimiciii-demo/"
        "mimic-iii-clinical-database-demo-1.4.zip"
    )

    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    with httpx.Client(
        headers={
            "User-Agent": browser_ua,
            "Accept": "*/*",
        },
        follow_redirects=True,
    ) as client:
        # Resolve Synthea sample CSV bundle URL
        synthea_html = client.get(synthea_page, timeout=60.0).text
        synthea_csv_url = _pick_anchor_href(
            synthea_html,
            base_url=synthea_page,
            contains_text="Download: CSV",
        )
        direct_specs.append(
            DownloadSpec(
                name="Synthea sample patients (CSV)",
                url=synthea_csv_url,
                out_path=raw / "synthea" / _sanitize_filename(Path(synthea_csv_url).name or "synthea_sample_csv.zip"),
            )
        )

        # MIMIC-III demo ZIP: direct, stable download URL (no credentialing required).
        mimic_zip_url = mimic_demo_zip
        direct_specs.append(
            DownloadSpec(
                name="MIMIC-III demo (zip)",
                url=mimic_zip_url,
                out_path=raw / "mimiciii_demo" / _sanitize_filename(Path(mimic_zip_url).name or "mimiciii_demo.zip"),
            )
        )

        print(f"Downloads: {len(direct_specs)}")
        for spec in direct_specs:
            print(f"\n== {spec.name} ==")
            _download_file(client, spec.url, spec.out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
