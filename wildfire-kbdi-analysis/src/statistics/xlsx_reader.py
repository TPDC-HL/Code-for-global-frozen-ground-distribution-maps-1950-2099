"""Read-only .xlsx table loader implemented using Python standard library.

Not a general Excel engine: formula cells must contain cached values.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

_N="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_P="http://schemas.openxmlformats.org/package/2006/relationships"

def col_index(addr: str) -> int:
    match=re.match(r"([A-Z]+)",addr.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {addr}")
    n=0
    for ch in match.group(1):
        n=n*26+ord(ch)-ord("A")+1
    return n-1

def read_xlsx_grid(filename,*,sheet="Sheet1",max_row=None,max_col=None):
    """Read grid to 2D lists; empty cells become None."""
    filename=Path(filename)
    with ZipFile(filename) as z:
        wb=ET.fromstring(z.read("xl/workbook.xml"))
        relations=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap={e.attrib["Id"]:e.attrib["Target"]
                for e in relations.findall(f"{{{_P}}}Relationship")}
        sheets=wb.find(f"{{{_N}}}sheets")
        all_sheets=list(sheets.findall(f"{{{_N}}}sheet"))
        if isinstance(sheet,int):
            sh=all_sheets[sheet]
        else:
            sh=next((x for x in all_sheets if x.attrib.get("name")==sheet),None)
            if sh is None:
                raise KeyError(f"Sheet {sheet!r} not found")
        target=relmap[sh.attrib[f"{{{_R}}}id"]].replace(chr(92),"/")
        if target.startswith("/"):
            path=target.lstrip("/")
        elif target.startswith("xl/"):
            path=target
        else:
            path="xl/"+target
        path=str(Path(path))
        if path not in z.namelist():
            raise FileNotFoundError(f"Worksheet missing: {path}")
        shared=[]
        if "xl/sharedStrings.xml" in z.namelist():
            ss=ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall(f"{{{_N}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_N}}}t")))
        xml=ET.fromstring(z.read(path))
        rows={}
        used_cols=0
        for row in xml.findall(f".//{{{_N}}}sheetData/{{{_N}}}row"):
            rownum=int(row.attrib.get("r",len(rows)+1))
            if max_row is not None and rownum>max_row:
                continue
            cells={}
            for cell in row.findall(f"{{{_N}}}c"):
                addr=cell.attrib.get("r")
                if not addr:
                    continue
                col=col_index(addr)
                if max_col is not None and col>=max_col:
                    continue
                value_tag=cell.find(f"{{{_N}}}v")
                inline=cell.find(f"{{{_N}}}is")
                typ=cell.attrib.get("t","n")
                if typ=="s" and value_tag is not None and value_tag.text is not None:
                    value=shared[int(value_tag.text)]
                elif typ=="inlineStr" and inline is not None:
                    value="".join(t.text or "" for t in inline.iter(f"{{{_N}}}t"))
                elif typ=="b" and value_tag is not None:
                    value=value_tag.text=="1"
                elif typ in ("str","e","d") and value_tag is not None:
                    value=value_tag.text
                elif value_tag is not None and value_tag.text not in (None,""):
                    try:
                        numeric=float(value_tag.text)
                        value=int(numeric) if numeric.is_integer() and abs(numeric)<2**53 else numeric
                    except ValueError:
                        value=value_tag.text
                else:
                    value=None
                cells[col]=value
                used_cols=max(used_cols,col+1)
            rows[rownum]=cells
        nrows=max_row if max_row is not None else max(rows,default=0)
        ncols=max_col if max_col is not None else used_cols
        return [[rows.get(i,{}).get(j) for j in range(ncols)] for i in range(1,nrows+1)]
