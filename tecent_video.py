
import requests
import json
from pprint import pprint
import re
from tqdm import tqdm
import os
from pypinyin import lazy_pinyin

path = 'data'
if not os.path.exists(path):
    os.mkdir(path)

# Cookie理论上不需要修改，  如需更新，Cookie可以从 proxyhttp , 标头, cookie栏处复制获得， 代表了登录的账号信息，下载4看数据是必须要的。
# user-agent 应该也不需要修改， 如果是linux系统可能可以更新一下， 位于 proxyhttp,标头，User-Agent栏位。
# 模拟浏览器
headers = {
    'cookie': 'RK=Sak0PdTjfb; ptcz=da84fd1e2d1dff57398f856e226be7aba8c858461b20331bb07ce7e0f81617e5; eas_sid=510741U6c1N2i1a0L3d8d4P9X4; qq_domain_video_guid_verify=964f25776e2b5c33; _qimei_uuid42=189020f00051000a10a8759436b9039f42f786f633; pgv_info=ssid=s7964405800; pgv_pvid=9623622769; _qimei_h38=3270242310a8759436b9039f02000002518902; vversion_name=8.2.95; video_omgid=964f25776e2b5c33; appuser=0EBD96B02CA67545; cm_cookie=V1,110064&o7YCs09BNRgI&AQEBABoBLW98twDL_gLEm4xVtiWT9DcG25BZ&240902&240902; _qimei_fingerprint=8db687da6f763b0ed4a7660031041a29; _qpsvr_localtk=0.9442346823121939; _qimei_q32=387ac44a95997df2e5c8718449cabb8f; _qimei_q36=4552ed18c2288b2d08f58e10300013318509; ad_session_id=wensxb3b4t3t2; lv_play_index=44; o_minduid=XbIz37ji57LIvUH9yr9qoNf5OqBMXsa_; Lturn=104; LKBturn=632; qz_gdt=ushnkzqfaaafzlezvpaa; orderid=19159640856_1725272053; LPVLturn=793; LPLFturn=391; LPSJturn=136; LVINturn=405; LPHLSturn=97; LDERturn=573; LPDFturn=100; LPPBturn=276; LZTturn=413; ufc=r64_1_1725355987',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0'
}
# url地址: 抓包找到链接， 通常所有腾讯视频的url都是相同的
# 可以在proxyhttp的标头的请求URL中获取
url = 'https://vd6.l.qq.com/proxyhttp'

# 请求参数样例, 理论上整个腾讯适配的data都可以使用同一个样例，
# 如果无法下载数据或遇到问题，可以尝试把proxyhttp中m3u8的负载项复制到此处。
"""请求数据的格式， 每次下载需要更新！！！"""
data = {"buid":"vinfoad","vinfoparam":"charge=0&otype=ojson&defnpayver=3&spau=1&spaudio=0&spwm=1&sphls=2&host=v.qq.com&refer=https%3A%2F%2Fv.qq.com%2Fx%2Fcover%2Fmzc002007gqzd02%2Fw4100l37pl0.html&ehost=https%3A%2F%2Fv.qq.com%2Fx%2Fcover%2Fmzc002007gqzd02%2Fw4100l37pl0.html&sphttps=1&encryptVer=9.2&cKey=5viOKELapre12M1Orq2-LnCjnpb8Ocr0cPTfEiyxzEul_f4uOWcoVmJJR8Gv67M9PQ8AgQRTlmraCp7VHCeQghpmp7rG5tiHjLv_PnnatnPaZfOXktuBpd_Iz_kq7YDF2132-IIK4Z1lFjOj-NmZhE-NjjawCzIdF66cdsFdzz5jk70UOmynTHDptaxqIemxrSlkg-M_BbDaBoWwiX7uSsBkDK_qlv9BvC71CgXl1BaviS_BpuKq82bQx2On2AOQ35yHG6e0-dJ37Szr2rFp56lxzzDtPFKhwYCtsQcaQ5txIefj4bzawfC8kP_trJ4QPNUngfY34pXVSaSl29Zyy7907ICoeOJZY6b_P75Em4nh9wpcMyYUi6FBXEYUygJzG6ZvZvAfPbh53IIv5Pw-jOnmfdKVqsXRolrT0Nl2wt9dw9beLkLMhqvyqC_l9KhBN9Ckx1SbI55SRAYzom_cnBFJ9PaO0U8USoK8Gdj6YYHZA_izkZfoWZswAmOHQCIXX0HzMj7U_Nw_yygDklmPK_cuKZCgXXwxDrgYOivBbEPzynwXMq0zNiXkLWAEBAQEejbk6Q&clip=4&guid=964f25776e2b5c33&flowid=24f4368739b51b0285c190e2b6bf1f48&platform=10201&sdtfrom=v1010&appVer=1.34.7&unid=&auth_from=&auth_ext=&vid=w4100l37pl0&defn=uhd&fhdswitch=0&dtype=3&spsrt=2&tm=1725358435&lang_code=0&logintoken=%7B%22access_token%22%3A%2284_-TZm-4KO4HEgdv9YYfPxZB4VLX9Clq1eSxGAYcNsAlm_VhweBEhagYm_xzADiNOUnUBKDaT4bCghl6Q9vuTg8V3B62vqSaZEsG6S0D8Iabc%22%2C%22appid%22%3A%22wx5ed58254bc0d6b7f%22%2C%22vusession%22%3A%22u4g3KkdvBv_BLYgja4j_Lw.M%22%2C%22openid%22%3A%22ox8XOvnN72OJ1baZ8y4FROpAHpLc%22%2C%22vuserid%22%3A%221312055853%22%2C%22video_guid%22%3A%22964f25776e2b5c33%22%2C%22main_login%22%3A%22wx%22%7D&spvvpay=1&spadseg=3&spvvc=3&spav1=15&hevclv=28&spsfrhdr=0&spvideo=0&spm3u8tag=67&spmasterm3u8=3&drm=296","sspAdParam":"{\"ad_scene\":1,\"pre_ad_params\":{\"ad_scene\":1,\"user_type\":2,\"video\":{\"base\":{\"vid\":\"w4100l37pl0\",\"cid\":\"mzc002007gqzd02\"},\"is_live\":false,\"type_id\":1,\"referer\":\"https://v.qq.com/channel/movie/list?filter_params=sort%3D75&page_id=channel_list_second_page&channel_id=100173\",\"url\":\"https://v.qq.com/x/cover/mzc002007gqzd02/w4100l37pl0.html\",\"flow_id\":\"24f4368739b51b0285c190e2b6bf1f48\",\"refresh_id\":\"4b6d500ec35ad9b11045047176f9292a_1725358121\",\"fmt\":\"fhd\"},\"platform\":{\"guid\":\"964f25776e2b5c33\",\"channel_id\":0,\"site\":\"web\",\"platform\":\"in\",\"from\":0,\"device\":\"pc\",\"play_platform\":10201,\"pv_tag\":\"film_qq_com|channel\",\"support_click_scan_integration\":true,\"qimei32\":\"387ac44a95997df2e5c8718449cabb8f\"},\"player\":{\"version\":\"1.34.7\",\"plugin\":\"3.5.21\",\"switch\":1,\"play_type\":\"0\"},\"token\":{\"type\":2,\"vuid\":1312055853,\"vuser_session\":\"u4g3KkdvBv_BLYgja4j_Lw.M\",\"app_id\":\"wx5ed58254bc0d6b7f\",\"open_id\":\"ox8XOvnN72OJ1baZ8y4FROpAHpLc\",\"access_token\":\"84_-TZm-4KO4HEgdv9YYfPxZB4VLX9Clq1eSxGAYcNsAlm_VhweBEhagYm_xzADiNOUnUBKDaT4bCghl6Q9vuTg8V3B62vqSaZEsG6S0D8Iabc\"},\"req_extra_info\":{\"now_timestamp_s\":1725358435,\"ad_frequency_control_time_list\":{\"full_pause_feedback_successive\":{\"ad_frequency_control_time_list\":[1725265980]},\"full_pause_short_bid_forbid_cid\":{\"ad_frequency_control_time_list\":[1725267789]},\"full_pause_short_bid_forbid_vid\":{\"ad_frequency_control_time_list\":[1725267789]},\"full_pause_feed_back_bid\":{\"ad_frequency_control_time_list\":[1725267791]},\"full_pause_feedback_bid_successive\":{\"ad_frequency_control_time_list\":[1725267791]},\"full_pause_short_vip\":{\"ad_frequency_control_time_list\":[1725358107]}}},\"extra_info\":{}}}","adparam":"adType=preAd&vid=w4100l37pl0&sspKey=owwl"}

# 可以在 00_gzc.....后缀的 M3U8 中的标头的请求URL中获取， 需要包括斜杠”/“
"""下载连接的前缀， 每次下载需要更新！！！"""
ts_url_prefix = "https://87f6c6cfb591a3c2ce59ca612f9300ac.v.smtcdns.com/vipts.tc.qq.com/AmvtOMASEaPRbdBk-pWBaz_vWA2iLZ4I30FhRvjb4gCU/B_tRCdt2L6hl1ezG-aht1_p5NFvyXRctl-HU4O9NK3NMDSmaOVoJqwRvRqDldWh1xC/svp_50112/j1-5TVrbCXEtl4EqG7iaOwReFv8zLcauvCHMtd-T5lO5fSJ0M8eA6oDQQbci1qIwFH9ZsPBOgWDM75ByktzyBs4elZwmV20Mb7hlFPLWYEBwnJSVcfpeEEcgvreGXVALLMnedujCpaTi5elzdp2AS5hyzk6c7RlxcRHOSfJ7fXtrOsVdXmRwKeWllW1muJw_F1N0ql7p6Y54O8Yepk1jHRZpJjKN1SbRbrJ4RwaeQUuNGmbea36ftw/"


# 发送请求
response = requests.post(url=url, json=data, headers=headers)
"""获取数据: 获取服务器返回响应数据"""
print("response is ", response)
# 获取响应json数据
json_data = response.json()
"""解析数据: 提取我们需要的数据内容"""
vinfo = json_data['vinfo']
# 把json字符串转成json字典
info_json = json.loads(vinfo)
pprint(info_json)
# 提取m3u8链接
m3u8_url = info_json['vl']['vi'][0]['ul']['ui'][0]['url']
# 提取视频标题
title = info_json['vl']['vi'][0]['ti']

# 转换成英文拼音，防止后续读取出错

pinyin_list = lazy_pinyin(title)
title_pinying = ''.join(word.capitalize() for word in pinyin_list)

print(f'正在下载{title}, PingYing文件名为：{title_pinying} 请稍后...')

# 对于 m3u8链接 发送请求, 获取数据内容
m3u8 = requests.get(url=m3u8_url).text
# 提取所有ts链接
ts_list = re.findall(',\n(.*?)\n#', m3u8)
print(ts_list)

# for循环遍历, 提取列表里面的元素
for ts in tqdm(ts_list):
#     # 构建完整的ts链接
    ts_url = ts_url_prefix + ts
    """保存数据"""
    # 对于视频片段链接发送请求, 获取二进制数据
    ts_content = requests.get(url=ts_url).content
    # 保存数据
    with open('data/' + title_pinying + '.mp4', mode='ab') as f:
        # 写入数据
        f.write(ts_content)
print(f'{title}, 文件{title_pinying}.mp4 下载完成!!!')