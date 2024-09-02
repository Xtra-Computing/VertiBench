import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data/real/satellite/uncompressed')
    parser.add_argument('-o', '--output_dir', type=str, default='data/real/satellite/clean')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # This is a list of AOIs that either
    # 1. have no high-res image
    # 2. have no low-res image
    errorlist = [
        "UNHCR-COGs026823",
        "UNHCR-UKRs002527",
        "Landcover-1237604",
        "Landcover-1155257",
        "Landcover-1156642",
        "Landcover-774737",
        "Landcover-763170",
        "Landcover-775854",
        "UNHCR-BFAs004227",
        "Landcover-1219520",
        "Landcover-1188668",
        "Landcover-770906",
        "Landcover-1148426",
        "UNHCR-CMRs032284",
        "UNHCR-GNBs001117",
        "Landcover-73253",
        "Landcover-1246064",
        "Landcover-775904",
        "Landcover-1288052",
        "Landcover-485752",
        "Landcover-1175132",
        "UNHCR-CAFs033240",
        "Landcover-1198471",
        "Landcover-1293888",
        "Landcover-1230525",
        "Landcover-1241829",
        "Landcover-555595",
        "Landcover-459218",
        "Landcover-491311",
        "Landcover-1280733",
        "UNHCR-BFAs004488",
        "Landcover-496866",
        "Landcover-775398",
        "Landcover-1111029",
        "UNHCR-NGAs035508",
        "Landcover-1119043",
        "Landcover-776473",
        "Landcover-777209",
        "Landcover-1149712",
        "Landcover-772811",
        "Landcover-488862",
        "Landcover-1260028",
        "Landcover-1596540",
        "UNHCR-SLEs002046",
        "Landcover-1166403",
        "UNHCR-NGAs035978",
        "UNHCR-IRQs010053",
        "UNHCR-CMRs004022",
        "UNHCR-NGAs035805",
        "Landcover-772258",
        "Landcover-1347210",
        "UNHCR-PAKs003492",
        "Landcover-769816",
        "UNHCR-BFAs004511",
        "Landcover-1168954",
        "Landcover-1183915",
        "Landcover-777291",
        "UNHCR-NGAs026841",
        "Landcover-774795",
        "UNHCR-SSDs003967",
        "UNHCR-NGAs037066",
        "UNHCR-BFAs004460",
        "Landcover-1255307",
        "Landcover-1229102",
        "Landcover-693606",
        "UNHCR-TCDs015426",
        "ASMSpotter-1-1-1",
    ]

    # Scan the subdirectories of data_dir, if the subdirectory name is not in errorlist, then create a soft symbolic
    # link to the subdirectory in data_dir in output_dir
    for subdir in os.listdir(args.data_dir):
        if subdir not in errorlist:
            os.symlink(os.path.abspath(os.path.join(args.data_dir, subdir)),
                       os.path.abspath(os.path.join(args.output_dir, subdir)))


