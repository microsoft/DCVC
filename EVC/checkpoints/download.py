import urllib.request


def download_one(url, target):
    urllib.request.urlretrieve(url, target)


def main():
    urls = {
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211227&authkey=AD8e586WrFlT6IE': 'EVC_LL.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211225&authkey=AOOYBdkfEmZ9rTo': 'EVC_LM_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211226&authkey=ADp_pN4gvxbHMrw': 'EVC_LS_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211228&authkey=AHCLXyxrm3UdXxU': 'EVC_ML_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211229&authkey=AGT8gpE50lHHixI': 'EVC_MM_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211230&authkey=ABwOafGhqBQcT9I': 'EVC_SL_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211231&authkey=ANrIn85RgtBH2wM': 'EVC_SS_MD.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211233&authkey=AC8tZbxQdbJDXCU': 'Scale_EVC_SL_MDRRL.pth.tar',
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211232&authkey=AAy8Q8QMM0dUxKg': 'Scale_EVC_SS_MDRRL.pth.tar',
    }
    for url in urls:
        target = urls[url]
        print("downloading", target)
        download_one(url, target)
        print("downloaded", target)


if __name__ == "__main__":
    main()
