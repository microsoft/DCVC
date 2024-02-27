import urllib.request


def download_one(url, target):
    urllib.request.urlretrieve(url, target)


def main():
    urls = {
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211494&authkey=!AOxzcrEFT_h-iCk': 'cvpr2024_image.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211493&authkey=!AFxYv6oK1o6GrZc': 'cvpr2024_video.pth.tar',
    }
    for url in urls:
        target = urls[url]
        print("downloading", target)
        download_one(url, target)
        print("downloaded", target)


if __name__ == "__main__":
    main()
