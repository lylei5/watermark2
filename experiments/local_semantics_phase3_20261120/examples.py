"""Illustrate all independently confirmed replacement pairs accepted by method C."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE=Path(__file__).resolve().parent


def main():
    with (HERE/'subject_scores.csv').open() as f:
        cases=[r for r in csv.DictReader(f) if r['cohort']=='test_public' and r['C_pass']=='1']
    with (HERE/'masks/mask_sizes.csv').open() as f:
        coverage={int(r['item']):float(r['masked_fraction']) for r in csv.DictReader(f)}
    tile=192
    sheet=Image.new('RGB',(2*(2*tile+18),((len(cases)+1)//2)*(tile+76)),'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for j,r in enumerate(cases):
        i=int(r['item'])
        folder=HERE/'additional' if i>=100 else HERE.parent/'rgb_moments_phase2_20261120/author'
        x,y=(j%2)*(2*tile+18),(j//2)*(tile+76)
        for k,(role,title) in enumerate([('ref','Registered'),('output','Edited')]):
            with Image.open(folder/f'replacement_{i}_{role}.jpg') as im:
                sheet.paste(ImageOps.pad(im.convert('RGB'),(tile,tile),color='white'),(x+k*tile,y+24))
            draw.text((x+k*tile+2,y+2),title,font=font,fill='black')
        draw.text((x+2,y+tile+30),f"Dev {i}   Mask {coverage[i]:.1%}   C distance {float(r['C_score']):.3f}",
                  fill='black',font=font)
    sheet.save(HERE/'missed_pairs.jpg',quality=91)


if __name__=='__main__':
    main()
