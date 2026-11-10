#!/usr/bin/env python3
"""Exact OLS/Pearson significance for the 12 CALM series in Thawing_Depths.xlsx.

Reproduces the statistics shown in Fig. 2 and appends an unobtrusive,
precisely typeset significance table to the unmodified source image.

Usage:
  python analyze_ALT_significance.py --input Thawing_Depths.xlsx --figure 'Fig. 2.png' --output-dir .

Notes:
  - Input contains ALT observations only. Pearson r is ALT vs calendar year,
    NOT ALT vs seasonal air temperature.
  - Statistical significance is two-sided OLS/Pearson with n-2 degrees of freedom.
  - Tests are unadjusted for serial correlation; FDR-adjusted p values are
    reported in the CSV as an additional diagnostic, not used to mark figure.
"""
import argparse
import csv
from pathlib import Path
import math

import numpy as np
from scipy.stats import linregress, pearsonr, t
from statsmodels.stats.multitest import multipletests
from artifact_tool import Blob, SpreadsheetFile
from PIL import Image, ImageDraw, ImageFont

SITE_ORDER = ['U2', 'U13', 'CA7', 'C18', 'S2', 'CH1',
              'R5', 'K0', 'M1', 'CN2', 'R9', 'R18']


def load_alt(path):
    wb = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet = wb.worksheets.get_item('Sheet1')
    cells = sheet.get_range('A1:M36').values
    names = [str(s) if s is not None else '' for s in cells[1]]
    result = {}
    for site in SITE_ORDER:
        col = names.index(site)
        pairs = [(float(row[0]), float(row[col]))
                 for row in cells[2:]
                 if isinstance(row[0], (int, float))
                 and isinstance(row[col], (int, float))
                 and math.isfinite(float(row[col]))]
        if len(pairs) < 3:
            raise ValueError(f'Insufficient data for {site}')
        year = np.array([xy[0] for xy in pairs])
        alt = np.array([xy[1] for xy in pairs])
        result[site] = year, alt
    return result


def ols_test(year, alt):
    lr = linregress(year, alt)
    r, p_r = pearsonr(year, alt)
    n = len(year)
    df = n - 2
    t_stat = lr.slope / lr.stderr
    p_t = 2*t.sf(abs(t_stat), df)
    assert np.isclose(p_t, lr.pvalue, rtol=1e-8, atol=1e-14)
    assert np.isclose(p_r, lr.pvalue, rtol=1e-8, atol=1e-14)
    tcrit = t.ppf(0.975, df)
    return dict(n=n, year_start=int(min(year)), year_end=int(max(year)),
                slope_cm_per_yr=lr.slope, intercept=lr.intercept,
                r=r, t=t_stat, df=df, p=p_t, slope_se=lr.stderr,
                slope_CI95_low=lr.slope-tcrit*lr.stderr,
                slope_CI95_high=lr.slope+tcrit*lr.stderr)


def p_str(p):
    return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'


def font(size, bold=False):
    base = '/usr/share/fonts/truetype/dejavu/'
    return ImageFont.truetype(base + ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), size)


def append_significance_strip(image_path, entries, output_path):
    """Preserve every pixel of original plot, append exact P and n table."""
    original = Image.open(image_path).convert('RGB')
    width, height = original.size
    top_pad = 10
    hdr_h, row_h, bottom_pad = 54, 52, 22
    extra = top_pad + hdr_h + 2*row_h + bottom_pad
    output = Image.new('RGB', (width, height+extra), '#FFFFFF')
    output.paste(original, (0, 0))
    draw = ImageDraw.Draw(output)
    xleft = 50
    colw = (width-2*xleft)/6
    y0 = height + top_pad
    draw.line((xleft,y0,width-xleft,y0), fill='#444444', width=2)
    draw.text((xleft+4, y0+11),
              'Trend significance: two-sided OLS t-test (ALT vs year);  * P < 0.05;  ns P >= 0.05',
              font=font(23, bold=True), fill='#1f2937')
    for idx, site in enumerate(SITE_ORDER):
        stat = entries[site]
        row, col = divmod(idx,6)
        x = round(xleft + col*colw)
        y = y0 + hdr_h + row*row_h
        # light, neutral row cells with stable site order matching graph
        if col % 2 == 0:
            draw.rectangle((x,y,x+colw-4,y+row_h-5), fill='#f1f5f9')
        mark = '*' if stat['p'] < .05 else 'ns'
        label = f"{site}{mark}:  n = {stat['n']},  {p_str(stat['p'])}"
        draw.text((x+6,y+12), label, font=font(18,bold=False), fill='#152337')
    draw.text((xleft, height+extra-18),
              'Slopes shown in the original panels are in cm yr-1; P values refer to year-versus-ALT trends (nominal two-sided tests).',
              font=font(13), fill='#52525b')
    output.save(output_path, dpi=(300,300), optimize=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('Thawing_Depths.xlsx'))
    parser.add_argument('--figure', type=Path, default=Path('Fig. 2.png'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=load_alt(args.input)
    stats={site: ols_test(*data[site]) for site in SITE_ORDER}
    pval=np.array([stats[s]['p'] for s in SITE_ORDER])
    adjusted=multipletests(pval, method='fdr_bh')[1]
    for site, padj in zip(SITE_ORDER, adjusted):
        stats[site]['p_FDR_BH_12sites'] = float(padj)
        stats[site]['significant_P_lt_0_05'] = stats[site]['p'] < .05

    outfile=args.output_dir/'ALT_12sites_significance.csv'
    fieldnames=['Site','n','year_start','year_end','slope_cm_per_yr','slope_se',
                'slope_CI95_low','slope_CI95_high','r','t','df','p',
                'p_FDR_BH_12sites','significant_P_lt_0_05']
    with open(outfile,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for site in SITE_ORDER:
            writer.writerow({'Site':site,**{k:stats[site][k] for k in fieldnames[1:]}})

    figout=args.output_dir/'Fig2_ALT_significance_annotated.png'
    append_significance_strip(args.figure,stats,figout)
    print('SITE\tN\tPERIOD\tOLS slope(cm/yr)\tr\tt\tP\tFDR q')
    for s in SITE_ORDER:
        z=stats[s]
        print(f"{s}\t{z['n']}\t{z['year_start']}-{z['year_end']}\t{z['slope_cm_per_yr']:+.3f}\t{z['r']:+.3f}\t{z['t']:.3f}\t{z['p']:.7g}\t{z['p_FDR_BH_12sites']:.7g}")
    print('Files:', outfile, figout)

if __name__ == '__main__':
    main()