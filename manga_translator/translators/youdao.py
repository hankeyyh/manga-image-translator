
# -*- coding: utf-8 -*-
import os
import uuid
import hashlib
import time
import aiohttp
import time

from .common import CommonTranslator, InvalidServerResponse, MissingAPIKeyException

def sha256_encode(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

class YoudaoTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh-CHS',
        'JPN': "ja",
        'ENG': 'en',
        'KOR': 'ko',
        'VIN': 'vi',
        'CSY': 'cs',
        'NLD': 'nl',
        'FRA': 'fr',
        'DEU': 'de',
        'HUN': 'hu',
        'ITA': 'it',
        'POL': 'pl',
        'PTB': 'pt',
        'ROM': 'ro',
        'RUS': 'ru',
        'ESP': 'es',
        'TRK': 'tr',
        'THA': 'th',
        'IND': 'id'
    }
    _API_URL = 'https://openapi.youdao.com/api'

    def __init__(self):
        super().__init__()
        if not self.get_youdao_keys() or not self.get_youdao_secret_key():
            raise MissingAPIKeyException('Please set the YOUDAO_APP_KEY and YOUDAO_SECRET_KEY environment variables before using the youdao translator.')

    async def _translate(self, from_lang, to_lang, queries):
        youdao_app_key = self.get_youdao_keys()
        youdao_secret_key = self.get_youdao_secret_key()
        if not youdao_app_key or not youdao_secret_key:
            raise MissingAPIKeyException('Please set the YOUDAO_APP_KEY and YOUDAO_SECRET_KEY environment variables before using the youdao translator.')

        data = {}
        query_text = '\n'.join(queries)
        data['from'] = from_lang
        data['to'] = to_lang
        data['signType'] = 'v3'
        curtime = str(int(time.time()))
        data['curtime'] = curtime
        salt = str(uuid.uuid1())
        signStr = youdao_app_key + self._truncate(query_text) + salt + curtime + youdao_secret_key
        sign = sha256_encode(signStr)
        data['appKey'] = youdao_app_key
        data['q'] = query_text
        data['salt'] = salt
        data['sign'] = sign
        #data['vocabId'] = "您的用户词表ID"

        result = await self._do_request(data)
        result_list = []
        if "translation" not in result:
            raise InvalidServerResponse(f'Youdao returned invalid response: {result}\nAre the API keys set correctly?')
        for ret in result["translation"]:
            result_list.extend(ret.split('\n'))
        return result_list

    def _truncate(self, q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    async def _do_request(self, data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        async with aiohttp.ClientSession() as session:
            async with session.post(self._API_URL, data=data, headers=headers) as resp:
                return await resp.json()

    def get_youdao_keys(self):
        # predict 时作为参数传入，再写入env，从env读取保证读到最新值
        return os.getenv('YOUDAO_APP_KEY') 

    def get_youdao_secret_key(self):
        # predict 时作为参数传入，再写入env，从env读取保证读到最新值
        return os.getenv('YOUDAO_SECRET_KEY') 
