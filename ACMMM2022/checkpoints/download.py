import urllib.request


def download_one(url, target):
    urllib.request.urlretrieve(url, target)


def main():
    urls = {
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211220&authkey=AMRg1W3PVt_F3yc': 'acmmm2022_image_psnr.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211219&authkey=ACJnPOPf1ntw_w0': 'acmmm2022_image_ssim.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211217&authkey=AKpdgXQtvs-OxRs': 'acmmm2022_video_psnr.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211218&authkey=ANxapLv3PcCJ4Vw': 'acmmm2022_video_ssim.pth.tar',
    }
    for url in urls:
        target = urls[url]
        print("downloading", target)
        download_one(url, target)
        print("downloaded", target)


if __name__ == "__main__":
    main()
