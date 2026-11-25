# Source PDFs

The annual reports are **not committed**. They are public documents, but they are
large and their redistribution terms are unclear. Only the derived index is
committed.

Download the Indonesian version of each report into this folder with the exact
names below, then run `python -m rag.ingest`.

| File | Bank | Year | Pages | SHA-256 |
|---|---|---|---|---|
| `BBCA_2023.pdf` | PT Bank Central Asia Tbk | 2023 | 747 | `4c23366b1732e992b51ed0ffbbde47e509dfb99a17ff1fef440b676c33c3f744` |
| `BBCA_2024.pdf` | PT Bank Central Asia Tbk | 2024 | 714 | `049733d9a21bbd4db6023759e8138b2d54db5e883d29d521da4c21b1e08dbef4` |
| `BBRI_2023.pdf` | PT Bank Rakyat Indonesia (Persero) Tbk | 2023 | 1182 | `8e2ae397da402bcda2af3c787cc18d8f0e53f7fc450970c700d4dd2f9efa1f6e` |
| `BBRI_2024.pdf` | PT Bank Rakyat Indonesia (Persero) Tbk | 2024 | 1257 | `cf012a5af0d766c4e1991bb74e2f4f0247f512aa1480418290e14143b23ec5c8` |
| `BMRI_2023.pdf` | PT Bank Mandiri (Persero) Tbk | 2023 | 1514 | `a7b2eb73e054e3c93e9214a2a28d4cf516caf56713be4015dc7c098320d28801` |
| `BMRI_2024.pdf` | PT Bank Mandiri (Persero) Tbk | 2024 | 1726 | `00b36dd77a59f3d3a1eac1dd5dc259e43c66e6016e91d453a1a011fef1865a24` |
| `BBNI_2023.pdf` | PT Bank Negara Indonesia (Persero) Tbk | 2023 | 1432 | `25cdde3a988804f464cf375718afb6142caaab68050adbdfb41d4b9f1b957410` |
| `BBNI_2024.pdf` | PT Bank Negara Indonesia (Persero) Tbk | 2024 | 1540 | `e860f463fe63998a2ed3d86b536b44bd4e445be7c6632c923c88f1104d5c9cc1` |
| `BRIS_2023.pdf` | PT Bank Syariah Indonesia Tbk | 2023 | 718 | `01e73ed711dba28444bc136ca2747dd29e78fc073d34b02798e8c99dffa34d41` |
| `BRIS_2024.pdf` | PT Bank Syariah Indonesia Tbk | 2024 | 684 | `4b70ee73c167b4b3acebae10e81ffe98d3fe37cb7189d30babae09693515afcb` |

## Official sources

- BBCA: <https://www.bca.co.id/id/tentang-bca/Hubungan-Investor/laporan-presentasi/Laporan-Tahunan>
  - 2024: `https://www.bca.co.id/-/media/Feature/Report/File/S8/Laporan-Tahunan/2025/20250306-BCA-AR-2024-ID.pdf`
  - 2023: `https://www.bca.co.id/-/media/Feature/Report/File/S8/Laporan-Tahunan/2024/20240215-buku-bca-ar-2023-ina-website.pdf`
- BBRI: <https://bri.co.id/report>
  - 2024: `https://bri.listedcompany.com/newsroom/AnnualReport2024-BBRI-att2.pdf`
  - 2023: `https://ir-bri.com/misc/AR/AR-2023-ID.pdf`
- BMRI: <https://www.bankmandiri.co.id/web/ir/annual-reports>
  - 2024: `https://www.bankmandiri.co.id/documents/38265486/0/%5BFINAL+-+2105%5D+LAPORAN+TAHUNAN+BMRI+2024+-+SINGLE.pdf/04bc7817-9ab4-4041-327b-06a732aa6c19?t=1747897840004`
  - 2023: `https://www.bankmandiri.co.id/documents/38265486/38265681/LAPORAN+TAHUNAN+BMRI+2023+%28FINAL%29+-+22+Mei+2024.pdf/60971633-3f12-bc88-e82c-8c8f6a73b1a7?t=1716371205973`
- BBNI: <https://www.bni.co.id/id-id/investor/laporan-keuangan>
  - 2024: `https://www.bni.co.id/Portals/1/BNI/Perusahaan/HubunganInvestor/Docs/AR-BNI-2024-IND.pdf`
  - 2023: `https://www.bni.co.id/Portals/1/BNI/Perusahaan/HubunganInvestor/Docs/AR-BOOK-IND-2023-Final.pdf`
- BRIS: <https://ir.bankbsi.co.id/annual_reports.html>
  - 2024: `https://ir.bankbsi.co.id/misc/AR/AR2024-ID.pdf`
  - 2023: `https://ir.bankbsi.co.id/misc/AR/AR2023-ID.pdf`

Verify a download with:

```bash
sha256sum data/raw/*.pdf
```
