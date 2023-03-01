import urllib.request


def download_one(url, target):
    urllib.request.urlretrieve(url, target)


def main():
    urls = {
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211259&authkey=AO_gFvTcYZUFd9U': 'cvpr2023_image_psnr.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211260&authkey=AFWlIyBB5PIudtw': 'cvpr2023_image_ssim.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211261&authkey=AOB9I7Jv25RbyGY': 'cvpr2023_image_yuv420_psnr.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211256&authkey=ACzRzK3XgbQxEyk': 'cvpr2023_video_psnr.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211258&authkey=AIRQMQyZqJWG15k': 'cvpr2023_video_ssim.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211257&authkey=AEQHk7O606IiqGA': 'cvpr2023_video_yuv420_psnr.pth.tar',
    }
    for url in urls:
        target = urls[url]
        print("downloading", target)
        download_one(url, target)
        print("downloaded", target)


if __name__ == "__main__":
    main()
