import sys
from pathlib import Path

covid_dir = Path(sys.argv[1]).resolve()
no_covid_dir = Path(sys.argv[2]).resolve()

covid_image_files = [file for file in covid_dir.iterdir() if file.name.endswith(".png")]
covid_size = sum((file.stat().st_size for file in covid_image_files))

no_covid_image_files = [file for file in no_covid_dir.iterdir() if file.name.endswith(".png")]
no_covid_size = sum((file.stat().st_size for file in no_covid_image_files))

print(f"""
COVID:
    Total number of images is: {len(covid_image_files):.2f}
    Total size of the images is: {covid_size/(1024**2):.2f} MB
--------------------------------------------------------------
NO_COVID:
    Total number of images is: {len(no_covid_image_files):.2f}
    Total size of the images is: {no_covid_size/(1024**2):.2f} MB
--------------------------------------------------------------
RATIOS
    Ratio of number of images no_covid/covid: {len(no_covid_image_files)/len(covid_image_files):.2f}
    Ratio of size of images no_covid/covid: {(no_covid_size/(1024**2))/(covid_size/(1024**2)):.2f} MB
""")
