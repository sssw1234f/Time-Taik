import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
import sys
sys.stdout.reconfigure(encoding='utf-8')

ARCHIVE_API_BASE = "http://archive.much.go.kr/openapi"

def fetch_test(search_keyword):
    endpoints = [
        f"{ARCHIVE_API_BASE}/nrms/listXml.do",
        f"{ARCHIVE_API_BASE}/publication/listXml.do",
    ]
    encoded_kw = urllib.parse.quote(search_keyword.encode("utf-8"))
    for ep in endpoints:
        url = f"{ep}?keyword={encoded_kw}&pageNo=1&numOfRows=10"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                content = resp.read()
                root = ET.fromstring(content)
                items = root.findall(".//item")
                print(f"[{ep.split('/')[-1]}] keyword={search_keyword} => {len(items)} items")
                for item in items[:3]:
                    img = item.findtext("imgUrl") or ""
                    name = item.findtext("name") or item.findtext("publicationNm") or ""
                    if img:
                        print(f"  title={name} | img={img[:60]}")
        except Exception as e:
            print(f"ERR {ep}: {e}")

for kw in ["만세운동", "아우내", "독립운동", "태극기", "안중근", "하얼빈", "윤봉길"]:
    fetch_test(kw)
    print()
