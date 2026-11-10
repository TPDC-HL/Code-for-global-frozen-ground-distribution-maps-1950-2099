"""Portable XLSX parser test with a synthetic workbook; no user data."""
from pathlib import Path
from zipfile import ZipFile
from xlsx_reader import col_index,read_xlsx_grid

def test_xlsx_reader_standard_library(tmp_path):
    xlsx=tmp_path/"tiny_synthetic.xlsx"
    with ZipFile(xlsx,"w") as z:
        z.writestr("xl/workbook.xml",'''<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>''')
        z.writestr("xl/_rels/workbook.xml.rels",'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>''')
        z.writestr("xl/sharedStrings.xml",'''<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Year</t></si><si><t>PF</t></si></sst>''')
        z.writestr("xl/worksheets/sheet1.xml",'''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2"><v>2001</v></c><c r="B2"><v>0.25</v></c></row></sheetData></worksheet>''')
    assert col_index("AB123")==27
    assert read_xlsx_grid(xlsx,sheet="Sheet1",max_row=3,max_col=2)==[
        ["Year","PF"],[2001,0.25],[None,None]]
