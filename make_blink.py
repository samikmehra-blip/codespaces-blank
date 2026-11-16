import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", help="Earlier SPHEREx FITS file")
    parser.add_argument("after", help="Later SPHEREx FITS file")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--hdu", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser.parse_args()


def read_fits(path, hdu):
    data, header = fits.getdata(path, hdu, header=True)
    return data.astype(np.float32), WCS(header).celestial, header


def display_frame(image, low, high):
    scaled = np.clip((image - low) / (high - low), 0, 1)
    scaled = np.arcsinh(8 * scaled) / np.arcsinh(8)
    return Image.fromarray(np.nan_to_num(scaled * 255).astype(np.uint8), mode="L")


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)

    before, before_wcs, before_header = read_fits(args.before, args.hdu)
    after, after_wcs, after_header = read_fits(args.after, args.hdu)

    target = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs")

    reference = Cutout2D(
        before,
        target,
        (args.size, args.size),
        wcs=before_wcs,
        mode="partial",
        fill_value=np.nan,
    )

    after_aligned, footprint = reproject_interp(
        (after, after_wcs),
        reference.wcs,
        shape_out=reference.data.shape,
    )

    valid_mask = (footprint > 0) & np.isfinite(reference.data)
    rows, cols = np.where(valid_mask)
    ymin, ymax = rows.min(), rows.max() + 1
    xmin, xmax = cols.min(), cols.max() + 1

    before_aligned = reference.data[ymin:ymax, xmin:xmax].astype(np.float32)
    after_aligned = after_aligned[ymin:ymax, xmin:xmax].astype(np.float32)
    footprint = footprint[ymin:ymax, xmin:xmax].astype(np.float32)

    low, high = np.nanpercentile(
        np.r_[before_aligned.ravel(), after_aligned.ravel()],
        (1, 99.5),
    )

    before_frame = display_frame(before_aligned, low, high)
    after_frame = display_frame(after_aligned, low, high)

    before_frame.save(args.output / "before.png")
    after_frame.save(args.output / "after.png")

    before_frame.save(
        args.output / "blink.gif",
        save_all=True,
        append_images=[after_frame],
        duration=700,
        loop=0,
    )

    np.savez_compressed(
        args.output / "aligned_pair.npz",
        before=before_aligned,
        after=after_aligned.astype(np.float32),
        difference=(after_aligned - before_aligned).astype(np.float32),
        footprint=footprint.astype(np.float32),
    )

    metadata = {
        "ra_deg": args.ra,
        "dec_deg": args.dec,
        "size_pixels": args.size,
        "before_file": str(args.before),
        "after_file": str(args.after),
        "before_date_obs": before_header.get("DATE-OBS"),
        "after_date_obs": after_header.get("DATE-OBS"),
        "display_percentiles": [1, 99.5],
        "aligned_coverage": float(np.mean(footprint > 0)),
        "alignment": "after image reprojected to before image WCS",
    }

    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"Created {args.output / 'blink.gif'}")


if __name__ == "__main__":
    main()