import requests
from bs4 import BeautifulSoup as bs


class proxy_rotator:

    def __init__(self, params= {4: "elite proxy", 6: "yes"}):

        #params: which proxies do you want
        #refresh: if proxies "run out", get new ones

        self.params = params
        self._get_proxy_list(params)
        self.index = 0
        self.bad_proxies = set()

    def refresh(self):

        self._get_proxy_list(self.params)
        self.index = 0

    def _get_proxy_list(self, params):

        r = requests.get("https://free-proxy-list.net/")
        bs_response = bs(r.text, "html.parser")
        table = bs_response.find("table")
        rows = table.findAll("tr")
        rows = [[col.text for col in row.findAll("td")] for row in rows]
        rows = rows[1:-1] #always empty list first and last

        self.proxy_list = []

        for row in rows:

            add = True

            for k,v in params.items():

                if row[k] != v:

                    add = False
                    break

            if add:

                if row[6] == "yes":

                    protocol = "https://"

                else:

                    protocol = "http://"

                self.proxy_list.append(protocol + row[0] + ":" + row[1])


    def get(self,url, params = {}, change_if_succesful = True, verbose = False):


        while self.index < len(self.proxy_list):

            proxy = self.proxy_list[self.index]
            if proxy in self.bad_proxies:

                self.index += 1
                continue

            try:

                if verbose:

                    print(f"Trying proxy....{proxy}")
                res = requests.get(url, proxies = {"https": proxy}, timeout = 5, params = params)
                if change_if_succesful:

                    self.index += 1

                return res

            except Exception as e:

                if verbose:

                    print(e)
                    print("Bad proxy....")
                self.bad_proxies.update([proxy])
                self.index += 1

        return None
