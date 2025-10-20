# کتابخانه‌های مورد نیاز را ایمپورت می‌کنیم
import pandas as pd
import numpy as np
import requests
from pandasdmx import Request
import matplotlib.pyplot as plt
import streamlit as st
from lxml import etree

# تنظیمات اولیه برای ظاهر نمودارها (فونت و اندازه)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11
})

# تابع کمکی برای واکشی فهرست نسخه‌های موجود (Edition) از ISTAT
@st.cache_data(show_spinner=False)
def get_istat_editions():
    """
    این تابع فهرست همه کدهای Edition موجود برای نرخ بیکاری ماهانه ISTAT (151_874) را برمی‌گرداند.
    از خروجی سری‌های ISTAT (با detail=serieskeysonly) استفاده می‌کند تا تمامی Editionها استخراج شوند.
    """
    url = "https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/151_874/M.IT.UNEM_R.N.9.Y15-74?detail=serieskeysonly"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        # در صورت وقوع خطا (مثلا timeout یا عدم دسترسی)، یک لیست خالی برمی‌گردانیم تا بعداً مدیریت شود
        return []
    # محتوای XML را با lxml تجزیه می‌کنیم تا کد Edition را استخراج کنیم
    editions = set()
    try:
        root = etree.fromstring(resp.content)
        # هر عنصر SeriesKey شامل چند مقدار (Value) است که یکی از آنها Edition است
        for sk in root.findall(".//SeriesKey"):
            for val in sk.findall(".//Value"):
                if val.get("concept") == "EDITION":
                    editions.add(val.get("value"))
        # مجموعه Editionها را به صورت لیست برمی‌گردانیم و مرتب می‌کنیم
        return sorted(list(editions))
    except Exception as e:
        return []

# تابع کمکی برای تعیین آخرین کد Edition (آخرین نسخه منتشرشده داده‌ها) از لیست Editionها
def get_latest_edition_code(editions):
    """
    با داشتن لیست کدهای Edition، آخرین کد (بر اساس تاریخ) را پیدا می‌کند.
    فرمت کدها: YYYYMxGy  (مثلاً 2025M10G2 یعنی 2 اکتبر 2025)
    روش: بر اساس سال، ماه و روز مرتب‌سازی می‌کند.
    """
    if not editions:
        return None
    # تابع تبدیل کد Edition به یک مقدار قابل مقایسه (سال، ماه، روز)
    def parse_edition_code(code):
        # جدا کردن بخش‌های سال، ماه و روز (با حذف "M" و "G")
        # مثال: "2025M10G2" -> سال=2025، ماه=10، روز=2
        try:
            year_str, rest = code.split("M", 1)
            year = int(year_str)
            month_str, day_str = rest.split("G", 1)
            month = int(month_str)
            day = int(day_str)
        except:
            year = month = day = -1
        return (year, month, day)
    # بر اساس سال/ماه/روز مرتب می‌کنیم و آخرین را انتخاب می‌کنیم
    latest = max(editions, key=lambda x: parse_edition_code(x))
    return latest

# تابع واکشی داده‌ها از ISTAT
@st.cache_data(show_spinner=False)
def fetch_from_istat(country_code, sex_code, age_code, adjust_code, start_period, end_period, edition_code=None):
    """
    تلاش برای واکشی سری زمانی نرخ بیکاری از ISTAT با استفاده از SDMX RESTful API.
    اگر edition_code مشخص نشده باشد، آخرین نسخه را به صورت خودکار پیدا می‌کند.
    country_code باید 'IT' باشد (داده‌های ISTAT فقط برای ایتالیا).
    sex_code (کد جنسیت)، age_code (کد گروه سنی)، adjust_code (کد تعدیل فصلی).
    """
    # اگر کد Edition داده نشده، تلاش می‌کنیم آخرین را پیدا کنیم
    ed_code = edition_code
    if ed_code is None:
        editions = get_istat_editions()
        ed_code = get_latest_edition_code(editions)
    # اگر هنوز هم Edition معلوم نیست، یک خطا ایجاد می‌کنیم
    if ed_code is None:
        raise ValueError("عدم تشخیص نسخه انتشار داده‌های ISTAT")
    # ساخت کلید SDMX بر اساس کدهای ابعاد 
    # ترتیب ابعاد: FREQ.REGION.DATA_TYPE.ADJUSTMENT.SEX.AGE.EDITION
    freq = "M"  # نرخ بیکاری ماهانه
    region = country_code  # برای ISTAT باید 'IT' باشد
    data_type = "UNEM_R"  # نرخ بیکاری
    adjustment = adjust_code  # 'N' یا 'Y' (و احتمالاً 'T' اگر وجود داشت)
    sex = sex_code  # '1', '2', '9'
    age = age_code  # مانند 'Y15-74' برای کل، یا 'Y15-24' و غیره
    edition = ed_code
    sdmx_key = f"{freq}.{region}.{data_type}.{adjustment}.{sex}.{age}.{edition}"
    # ساخت URL درخواست داده
    url = f"https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/151_874/{sdmx_key}"
    params = {}
    # اگر تاریخ شروع/پایان مشخص شده باشد، آنها را به‌صورت پارامتر SDMX (startPeriod, endPeriod) اضافه می‌کنیم
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    # درخواست داده به ISTAT
    resp = requests.get(url, params=params, timeout=15)
    # اگر پاسخ موفق نبود، خطا می‌دهیم تا به fallback برسیم
    resp.raise_for_status()
    # واکشی داده با pandasdmx برای سادگی تبدیل به DataFrame
    istat_req = Request('ISTAT')
    # تلاش می‌کنیم از pandasdmx برای دریافت داده استفاده کنیم (پانداسDMX خودش از SDMX-ML استفاده می‌کند)
    data_msg = istat_req.data('151_874', key={'FREQ': freq, 'REF_AREA': region, 'DATA_TYPE': data_type,
                                             'ADJUSTMENT': adjustment, 'SEX': sex, 'AGE': age, 'EDITION': edition},
                              params={'startPeriod': start_period, 'endPeriod': end_period})
    # تبدیل پیام داده به یک DataFrame پانداس
    data_series = None
    try:
        data_series = data_msg.to_pandas()
    except Exception as e:
        # اگر pandasdmx نتواند مستقیماً تبدیل کند، خودمان با xml parse می‌کنیم
        root = etree.fromstring(resp.content)
        obs_values = []
        obs_dates = []
        for obs in root.findall(".//Series/Obs"):
            # هر Obs دارای TimePeriod و ObsValue است
            time_elem = obs.find(".//ObsDimension")
            val_elem = obs.find(".//ObsValue")
            if time_elem is not None and val_elem is not None:
                time_val = time_elem.get("value")
                obs_val = val_elem.get("value")
                # تبدیل مقادیر به float (مقادیر ممکن است به صورت رشته باشند)
                try:
                    obs_val = float(obs_val)
                except:
                    obs_val = None
                obs_dates.append(time_val)
                obs_values.append(obs_val)
        if not obs_dates:
            raise ValueError("داده‌ای از ISTAT بازیابی نشد")
        data_series = pd.Series(data=obs_values, index=pd.to_datetime(obs_dates, format="%Y-%m"))
    # اگر Series برگردانده شد، آن را به DataFrame (یک ستون) تبدیل می‌کنیم
    df = pd.DataFrame({"Unemployment Rate": data_series})
    # مرتب‌سازی بر اساس تاریخ (اندیس)
    df = df.sort_index()
    return df

# تابع واکشی داده‌ها از Eurostat
@st.cache_data(show_spinner=False)
def fetch_from_eurostat(country_code, sex_code, age_code, adjust_code, start_period, end_period):
    """
    واکشی نرخ بیکاری ماهانه از Eurostat (دیتاست une_rt_m) با استفاده از pandasdmx.
    country_code کد کشور (مانند 'IT' برای ایتالیا)، sex_code ('T','M','F'), age_code ('TOTAL','Y_LT25','Y25-74'),
    adjust_code ('NSA','SA','TC').
    """
    estat_req = Request('ESTAT')
    # استفاده از pandasdmx برای دریافت داده، تعیین فیلترها
    # dataset: une_rt_m
    # ابعاد: unit, s_adj, sex, age, geo
    # واحد PC_ACT (درصد فعالان)
    # sex, age, adjust طبق انتخاب کاربر
    # geo طبق کشور
    key_filters = {
        'unit': 'PC_ACT',
        's_adj': adjust_code,
        'sex': sex_code,
        'age': age_code,
        'geo': country_code
    }
    params = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    data_msg = estat_req.data('une_rt_m', key=key_filters, params=params)
    # تبدیل به DataFrame (pandasdmx به صورت Series چندبعدی یا DataFrame چندشاخص برمی‌گرداند)
    data_series = data_msg.to_pandas()
    # data_series ممکن است یک Series چندشاخص (MultiIndex) باشد (اگر فقط یک سری باشد، خودش Series می‌شود)
    # ما فقط یک ترکیب sex/age/geo داریم، بنابراین باید یک سری ساده باشد. اگر DataFrame بود، آن را تک ستونه می‌کنیم.
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({"Unemployment Rate": data_series})
    else:
        # اگر نتیجه DataFrame بود (چندین سری)، فیلتر می‌کنیم فقط سری مورد نظر را.
        try:
            df = pd.DataFrame({"Unemployment Rate": data_series.xs((sex_code, age_code, country_code, adjust_code, "PC_ACT"), level=[2,3,4,1,0])})
        except:
            # اگر xs موفق نبود، مستقیماً اولین ستون را می‌گیریم
            df = pd.DataFrame(data_series)
            df.columns = ["Unemployment Rate"]
    # اندیس زمان را به تایم‌استمپ پایتون تبدیل می‌کنیم (اگر به صورت Period باشد)
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    # مرتب‌سازی تاریخ
    df = df.sort_index()
    return df

# تابع واکشی داده‌های کمکی (Auxiliary) از Eurostat
@st.cache_data(show_spinner=False)
def fetch_aux_from_eurostat(dataset_code, country_code, sex_code, age_code):
    """
    واکشی داده‌های کمکی از Eurostat. dataset_code می‌تواند 'lfsq_argan' (نرخ مشارکت) یا 'lfsq_agan' (جمعیت فعال) باشد.
    داده‌های LFS فصلی یا سالانه هستند. این تابع نزدیک‌ترین فرکانس را برمی‌گرداند (احتمالاً فصلی).
    sex_code, age_code بر اساس انتخاب کاربر.
    """
    estat_req = Request('ESTAT')
    # تنظیم فیلترها. فرض می‌کنیم اگر واحد لازم باشد، کتابخانه به‌صورت خودکار تنظیم می‌کند.
    key_filters = {
        'sex': sex_code,
        'age': age_code,
        'geo': country_code
    }
    # اگر dataset_code مربوط به جمعیت فعال باشد، احتمالاً واحد آن 'THS' (هزار نفر) است
    if dataset_code == 'lfsq_agan':
        key_filters['unit'] = 'THS'
    elif dataset_code == 'lfsq_argan':
        # برای نرخ مشارکت احتمالاً درصد جمعیت (ممکن است واحد 'PC_POP' باشد، بررسی می‌کنیم)
        key_filters['unit'] = 'PC_POP'  # درصد جمعیت در نیروی کار (تخمینی)
    try:
        data_msg = estat_req.data(dataset_code, key=key_filters)
    except Exception as e:
        # اگر درخواست اولیه با واحد تنظیم‌شده خطا داد، بدون unit دوباره تلاش می‌کنیم (ممکن است واحد ثابت باشد)
        key_filters.pop('unit', None)
        data_msg = estat_req.data(dataset_code, key=key_filters)
    data_series = data_msg.to_pandas()
    # تبدیل به DataFrame
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({dataset_code: data_series})
    else:
        # اگر سری‌های متعدد بود، سعی می‌کنیم فیلتر sex/age/geo را انتخاب کنیم
        try:
            df = pd.DataFrame(data_series.xs((sex_code, age_code, country_code), level=[1,2,3]))
            df.columns = [dataset_code]
        except:
            df = pd.DataFrame(data_series)
            df.columns = [dataset_code]
    # تبدیل اندیس دوره به تاریخ
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df = df.sort_index()
    return df

# تابع واکشی داده‌های کمکی از ISTAT
@st.cache_data(show_spinner=False)
def fetch_aux_from_istat(flow_id, country_code, sex_code, age_code, adjust_code):
    """
    واکشی داده کمکی از ISTAT برای flow_id داده‌شده (مثلاً 150_876 برای نرخ فعالیت، 150_873 برای جمعیت فعال).
    سایر کدهای ابعاد مشابه نرخ بیکاری پر می‌شوند.
    adjust_code برای تعیین تعدیل فصلی ('N' یا 'Y') استفاده می‌شود.
    """
    # data_type بستگی به flow دارد. به عنوان مثال:
    # برای 150_876 (نرخ مشارکت) ممکن است data_type = 'ACTIVITY_RATE' یا مشابه باشد.
    # برای 150_873 (جمعیت فعال) ممکن است 'LABOUR_FORCE' باشد.
    # برای سادگی، از flow_id استفاده می‌کنیم فرض بر این که فقط یک اندیکاتور دارد و نیازی به data_type نیست.
    freq = "M"
    region = country_code
    # تعیین data_type بر اساس flow_id:
    if flow_id == "150_876":
        data_type = "ACTIVITY_RATE"  # فرضی: نرخ فعالیت
    elif flow_id == "150_873":
        data_type = "LABOUR_FORCE"   # فرضی: جمعیت فعال
    else:
        data_type = None
    # اگر data_type تعریف شده باشد، در کلید استفاده می‌کنیم؛ در غیراینصورت خالی می‌گذاریم (احتمالاً flow فقط یک سری دارد)
    # تلاش می‌کنیم Edition آخر را نیز مانند نرخ بیکاری بگیریم:
    editions = get_istat_editions()  # ممکن است برای flowهای دیگر نیز Edition یکسان یا مشابه باشد (با نام متفاوت)
    ed_code = get_latest_edition_code(editions) if editions else None
    if ed_code is None:
        raise ValueError("عدم یافتن نسخه انتشار برای داده‌های کمکی ISTAT.")
    if data_type:
        sdmx_key = f"{freq}.{region}.{data_type}.{adjust_code}.{sex_code}.{age_code}.{ed_code}"
    else:
        # اگر data_type وجود ندارد (flow تک‌سری)، ممکن است ساختار کلید یکی کمتر بعد داشته باشد
        sdmx_key = f"{freq}.{region}.{adjust_code}.{sex_code}.{age_code}.{ed_code}"
    url = f"https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/{flow_id}/{sdmx_key}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    istat_req = Request('ISTAT')
    # آماده‌سازی فیلترهای pandasdmx
    key_filters = {
        'FREQ': freq,
        'REF_AREA': region,
        'SEX': sex_code,
        'AGE': age_code,
        'ADJUSTMENT': adjust_code
    }
    if data_type:
        key_filters['DATA_TYPE'] = data_type
    data_msg = istat_req.data(flow_id, key=key_filters, params={})
    data_series = data_msg.to_pandas()
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({flow_id: data_series})
    else:
        df = pd.DataFrame(data_series)
        df.columns = [flow_id]
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df = df.sort_index()
    return df

# ---- ساخت رابط کاربری Streamlit ----

st.set_page_config(page_title="نرخ بیکاری - ISTAT/Eurostat", layout="wide")

st.title("نمایش نرخ بیکاری ماهانه (意)")

# فهرست کشورها (کدهای geo) برای انتخاب کاربر
countries = {
    "ایتالیا": "IT",
    "آلمان": "DE",
    "فرانسه": "FR",
    "اسپانیا": "ES",
    "اتحادیه اروپا": "EU27_2020"
}
country_name = st.sidebar.selectbox("کشور / منطقه:", list(countries.keys()), index=0)
country_code = countries[country_name]

# انتخاب جنسیت
sex_options = {"کل جمعیت": "9", "مردان": "1", "زنان": "2"}
# (برای Eurostat معادل‌های آنها: 'T','M','F')
sex_label = st.sidebar.selectbox("جنسیت:", list(sex_options.keys()), index=0)
sex_code_istat = sex_options[sex_label]    # برای ISTAT
sex_code_eurostat = "T" if sex_code_istat == "9" else ("M" if sex_code_istat == "1" else "F")

# انتخاب گروه سنی
age_options = {
    "15-74 (کل نیروی کار)": "Y15-74",
    "15-24 (نرخ بیکاری جوانان)": "Y15-24",
    "25-74 (بزرگسالان)": "Y25-74",
    "15-34": "Y15-34",
    "35-49": "Y35-49",
    "50-64": "Y50-64",
    "50-74": "Y50-74",
    "15-64": "Y15-64"
}
age_label = st.sidebar.selectbox("گروه سنی:", list(age_options.keys()), index=0)
age_code_istat = age_options[age_label]
# تبدیل کد گروه سنی به فرمت Eurostat
# Eurostat فقط از 'TOTAL','Y_LT25','Y25-74' استفاده می‌کند (در صورت انتخاب سایر گروه‌ها، به نزدیک‌ترین تقریب می‌بریم)
if age_code_istat == "Y15-74":
    age_code_eurostat = "TOTAL"
elif age_code_istat in ["Y15-24", "Y15-34"]:
    age_code_eurostat = "Y_LT25"  # فرض: کمتر از 25
elif age_code_istat == "Y25-74":
    age_code_eurostat = "Y25-74"
elif age_code_istat == "Y15-64":
    age_code_eurostat = "TOTAL"  # Eurostat تعریف 15-64 در این داده ندارد، از کل 15-74 استفاده می‌کنیم
elif age_code_istat in ["Y35-49", "Y50-64", "Y50-74"]:
    age_code_eurostat = "TOTAL"  # Eurostat چنین تفکیکی در این دیتاست ندارد، کل را می‌گیریم
else:
    age_code_eurostat = "TOTAL"

# انتخاب نوع تعدیل فصلی
adjust_options = {"خام (بدون تعدیل)": "NSA", "تعدیل‌شده فصلی": "SA", "مولفه‌ی روند": "TC"}
adjust_label = st.sidebar.selectbox("تعدیل:", list(adjust_options.keys()), index=1)
adjust_code_eurostat = adjust_options[adjust_label]
# تبدیل به کد ISTAT ('N' یا 'Y' - مولفه روند در ISTAT موجود نیست)
if adjust_code_eurostat == "NSA":
    adjust_code_istat = "N"
elif adjust_code_eurostat == "SA":
    adjust_code_istat = "Y"
else:
    adjust_code_istat = "N"  # اگر TrendCycle انتخاب شود و ISTAT نداشته باشد، بر مبنای خام اقدام می‌کنیم

# انتخاب بازه تاریخ
start_year = st.sidebar.number_input("سال شروع:", min_value=2000, max_value=2030, value=2015)
start_month = st.sidebar.selectbox("ماه شروع:", list(range(1, 13)), index=0)
end_year = st.sidebar.number_input("سال پایان:", min_value=2000, max_value=2030, value=2025)
end_month = st.sidebar.selectbox("ماه پایان:", list(range(1, 13)), index=9)
start_period = f"{start_year}-{start_month:02d}"
end_period = f"{end_year}-{end_month:02d}"

# انتخاب متغیرهای کمکی برای واکشی
aux_participation = st.sidebar.checkbox("نرخ مشارکت اقتصادی (نرخ فعالیت)", value=False)
aux_active_pop = st.sidebar.checkbox("جمعیت فعال (نیروی کار)", value=False)

# با زدن یک دکمه، عملیات واکشی و نمایش انجام می‌شود
if st.sidebar.button("نمایش نتایج"):
    data_source = None  # منبع داده استفاده شده (ISTAT یا Eurostat یا 'sample')
    main_df = None      # دیتافریم اصلی نرخ بیکاری
    aux_dfs = {}        # فرهنگ دیتافریم‌های متغیرهای کمکی
    # تلاش برای واکشی از ISTAT (در صورت انتخاب ایتالیا و در دسترس بودن بعد Trend)
    use_istat = (country_code == "IT")
    # اگر کاربر TrendCycle را انتخاب کرده اما ISTAT این گزینه را ندارد، از Eurostat استفاده خواهیم کرد
    if adjust_code_eurostat == "TC" and country_code == "IT":
        # پیغام به کاربر که TrendCycle از Eurostat گرفته می‌شود
        st.info("مولفه روند برای سری ایتالیا مستقیماً از Eurostat واکشی می‌شود (در ISTAT موجود نیست).")
        use_istat = False
    if use_istat:
        try:
            main_df = fetch_from_istat(country_code, sex_code_istat, age_code_istat, adjust_code_istat, start_period, end_period)
            data_source = "ISTAT"
        except Exception as e:
            # در صورت هرگونه خطا، به Eurostat مراجعه می‌کنیم
            use_istat = False
    if not use_istat:
        try:
            main_df = fetch_from_eurostat(country_code, sex_code_eurostat, age_code_eurostat, adjust_code_eurostat, start_period, end_period)
            data_source = "Eurostat"
        except Exception as e:
            # اگر Eurostat هم شکست خورد، از داده نمونه استفاده می‌کنیم
            data_source = "sample"
            # ساخت داده نمونه ساده
            date_idx = pd.date_range(start=pd.to_datetime(start_period), end=pd.to_datetime(end_period), freq='M')
            # تولید مقادیر فرضی برای نرخ بیکاری (صرفاً جهت نمایش)
            sample_values = np.linspace(10, 5, num=len(date_idx)) + np.random.randn(len(date_idx)) * 0.5
            main_df = pd.DataFrame({"Unemployment Rate": sample_values}, index=date_idx)
            st.warning("عدم امکان واکشی داده‌های واقعی. نمایش داده نمونه.")
    # اگر main_df آماده است، نمایش نتایج
    if main_df is not None:
        # واکشی داده‌های کمکی در صورت انتخاب کاربر
        if aux_participation:
            if data_source == "ISTAT" and country_code == "IT":
                try:
                    aux_df = fetch_aux_from_istat("150_876", country_code, sex_code_istat, age_code_istat, adjust_code_istat)
                    aux_dfs["Activity Rate"] = aux_df
                except Exception as e:
                    # اگر ISTAT برای نرخ فعالیت خطا داد، سعی می‌کنیم Eurostat را امتحان کنیم
                    try:
                        aux_df = fetch_aux_from_eurostat("lfsq_argan", country_code, sex_code_eurostat, age_code_eurostat)
                        aux_dfs["Activity Rate"] = aux_df
                    except:
                        st.error("عدم موفقیت در واکشی نرخ مشارکت اقتصادی.")
            else:
                # اگر از Eurostat برای داده اصلی استفاده شده یا کشور غیر از ایتالیاست
                try:
                    aux_df = fetch_aux_from_eurostat("lfsq_argan", country_code, sex_code_eurostat, age_code_eurostat)
                    aux_dfs["Activity Rate"] = aux_df
                except:
                    st.error("عدم موفقیت در واکشی نرخ مشارکت اقتصادی.")
        if aux_active_pop:
            if data_source == "ISTAT" and country_code == "IT":
                try:
                    aux_df = fetch_aux_from_istat("150_873", country_code, sex_code_istat, age_code_istat, adjust_code_istat)
                    aux_dfs["Active Population"] = aux_df
                except Exception as e:
                    try:
                        aux_df = fetch_aux_from_eurostat("lfsq_agan", country_code, sex_code_eurostat, age_code_eurostat)
                        aux_dfs["Active Population"] = aux_df
                    except:
                        st.error("عدم موفقیت در واکشی جمعیت فعال.")
            else:
                try:
                    aux_df = fetch_aux_from_eurostat("lfsq_agan", country_code, sex_code_eurostat, age_code_eurostat)
                    aux_dfs["Active Population"] = aux_df
                except:
                    st.error("عدم موفقیت در واکشی جمعیت فعال.")
        # اگر داده اصلی از Eurostat باشد اما اولویت با ISTAT بوده (مثل کشور ایتالیا ولی ISTAT شکست خورده)، پیام اطلاع‌رسانی
        if data_source == "Eurostat" and country_code == "IT":
            st.info("داده‌ها از Eurostat واکشی شده‌اند (به‌عنوان جایگزین ISTAT).")
        # ایجاد نمودار
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(main_df.index, main_df["Unemployment Rate"], label="نرخ بیکاری", color='C0', linewidth=2)
        # اگر داده کمکی وجود دارد، آنها را نیز روی نمودار ترسیم می‌کنیم (با محور ثانویه در صورت نیاز)
        if aux_dfs:
            # اگر یک متغیر کمکی انتخاب شده باشد، می‌توانیم آن را روی محور راست نمایش دهیم برای مقیاس متفاوت
            # اگر دو متغیر باشند، هر دو را روی همان محور دوم یا یکی روی اول یکی روی دوم؟ 
            # برای سادگی، هر دو را هم‌مقیاس نمی‌کنیم، فقط رسم با برچسب
            for name, df_aux in aux_dfs.items():
                # تنظیم اینکه داده کمکی ممکن است در فرکانس فصلی باشد. با reindex به فرکانس ماهانه (پراکندن مقادیر) تبدیل می‌کنیم.
                df_aux_resampled = df_aux.copy()
                if df_aux_resampled.index.inferred_freq is None or df_aux_resampled.index.freq != 'M':
                    # مقادیر را فوروارد-فیل می‌کنیم تا به صورت ماهانه درآیند (هر مقدار فصلی به ماه‌های دوره تعمیم می‌یابد)
                    df_aux_resampled = df_aux_resampled.resample('M').ffill()
                # رسم
                ax.plot(df_aux_resampled.index, df_aux_resampled[df_aux_resampled.columns[0]], label=name, linestyle='--')
        ax.set_title(f"نرخ بیکاری ماهانه - {country_name} ({sex_label}, {age_label})")
        ax.set_xlabel("تاریخ")
        ax.set_ylabel("نرخ بیکاری (%)")
        ax.legend()
        st.pyplot(fig)
        # محاسبه و نمایش شاخص‌های آماری (آخرین مقدار، میانگین، تغییر ماهانه و سالانه)
        latest_value = main_df["Unemployment Rate"].iloc[-1]
        avg_value = main_df["Unemployment Rate"].mean()
        change_mom = None
        change_yoy = None
        if len(main_df) >= 2:
            change_mom = latest_value - main_df["Unemployment Rate"].iloc[-2]
        # تغییر سالانه (اختلاف نسبت به 12 ماه قبل در صورت وجود)
        if len(main_df) > 12:
            change_yoy = latest_value - main_df["Unemployment Rate"].iloc[-13]
        # نمایش به‌صورت متن
        st.subheader("شاخص‌های کلیدی:")
        cols = st.columns(4)
        cols[0].metric("آخرین مقدار", f"{latest_value:.2f}%")
        cols[1].metric("میانگین", f"{avg_value:.2f}%")
        if change_mom is not None:
            cols[2].metric("تغییر ماه قبل", f"{change_mom:+.2f} واحد")
        else:
            cols[2].text("تغییر ماه قبل: داده کافی نیست")
        if change_yoy is not None:
            cols[3].metric("تغییر سال قبل", f"{change_yoy:+.2f} واحد")
        else:
            cols[3].text("تغییر سال قبل: داده کافی نیست")
        # محاسبه و نمایش ضریب همبستگی اگر داده کمکی وجود دارد
        if aux_dfs:
            st.subheader("همبستگی با متغیرهای کمکی:")
            for name, df_aux in aux_dfs.items():
                # تراز زمانی دو سری (فقط در محدوده مشترک)
                combined = pd.DataFrame({"unemp": main_df["Unemployment Rate"]})
                # resample داده کمکی به ماهانه (مثل قبل)
                df_aux_resampled = df_aux.copy()
                if df_aux_resampled.index.inferred_freq is None or df_aux_resampled.index.freq != 'M':
                    df_aux_resampled = df_aux_resampled.resample('M').ffill()
                combined[name] = df_aux_resampled[df_aux_resampled.columns[0]]
                # حذف مقادیر گمشده
                combined = combined.dropna()
                if combined.shape[0] < 2:
                    st.write(f"محاسبه همبستگی برای {name}: داده کافی نیست.")
                else:
                    corr_val = combined["unemp"].corr(combined[name])
                    st.write(f"ضریب همبستگی بین نرخ بیکاری و **{name}**: {corr_val:.2f}")

# کتابخانه‌های مورد نیاز
import pandas as pd
import numpy as np
import requests
from pandasdmx import Request
import matplotlib.pyplot as plt
import streamlit as st
from lxml import etree

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11
})

# تابع کمکی: واکشی لیست نسخه‌های انتشار (Edition) از ISTAT
@st.cache_data(show_spinner=False)
def get_istat_editions():
    """
    فهرست تمام Editionهای موجود برای داده‌فلو 151_874 را واکشی می‌کند.
    """
    url = "https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/151_874/M.IT.UNEM_R.N.9.Y15-74?detail=serieskeysonly"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []
    editions = set()
    try:
        root = etree.fromstring(resp.content)
        for sk in root.findall(".//SeriesKey"):
            for val in sk.findall(".//Value"):
                if val.get("concept") == "EDITION":
                    editions.add(val.get("value"))
        return sorted(list(editions))
    except Exception:
        return []

# تابع کمکی: تعیین آخرین Edition (جدیدترین) از لیست کدها
def get_latest_edition_code(editions):
    if not editions:
        return None
    def parse_code(code):
        try:
            year_str, rest = code.split("M", 1)
            month_str, day_str = rest.split("G", 1)
            year = int(year_str)
            month = int(month_str)
            day = int(day_str)
        except:
            year = month = day = -1
        return (year, month, day)
    latest = max(editions, key=lambda x: parse_code(x))
    return latest

# تابع واکشی نرخ بیکاری از ISTAT
@st.cache_data(show_spinner=False)
def fetch_from_istat(country_code, sex_code, age_code, adjust_code, start_period, end_period, edition_code=None):
    # تعیین Edition آخر در صورت عدم ارائه
    ed_code = edition_code if edition_code else get_latest_edition_code(get_istat_editions())
    if ed_code is None:
        raise ValueError("عدم شناسایی نسخه انتشار داده‌های ISTAT")
    freq = "M"
    region = country_code
    data_type = "UNEM_R"
    adjustment = adjust_code
    sex = sex_code
    age = age_code
    edition = ed_code
    sdmx_key = f"{freq}.{region}.{data_type}.{adjustment}.{sex}.{age}.{edition}"
    url = f"https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/151_874/{sdmx_key}"
    params = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    istat_req = Request('ISTAT')
    data_msg = istat_req.data('151_874', key={
        'FREQ': freq, 'REF_AREA': region, 'DATA_TYPE': data_type,
        'ADJUSTMENT': adjustment, 'SEX': sex, 'AGE': age, 'EDITION': edition
    }, params={'startPeriod': start_period, 'endPeriod': end_period})
    try:
        data_series = data_msg.to_pandas()
    except Exception:
        root = etree.fromstring(resp.content)
        obs_dates, obs_values = [], []
        for obs in root.findall(".//Series/Obs"):
            time_elem = obs.find(".//ObsDimension")
            val_elem = obs.find(".//ObsValue")
            if time_elem is not None and val_elem is not None:
                obs_dates.append(time_elem.get("value"))
                try:
                    obs_values.append(float(val_elem.get("value")))
                except:
                    obs_values.append(None)
        if not obs_dates:
            raise ValueError("داده‌ای از ISTAT بازیابی نشد")
        data_series = pd.Series(obs_values, index=pd.to_datetime(obs_dates, format="%Y-%m"))
    df = pd.DataFrame({"Unemployment Rate": data_series}).sort_index()
    return df

# تابع واکشی نرخ بیکاری از Eurostat
@st.cache_data(show_spinner=False)
def fetch_from_eurostat(country_code, sex_code, age_code, adjust_code, start_period, end_period):
    estat_req = Request('ESTAT')
    key_filters = {
        'unit': 'PC_ACT',
        's_adj': adjust_code,
        'sex': sex_code,
        'age': age_code,
        'geo': country_code
    }
    params = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    data_msg = estat_req.data('une_rt_m', key=key_filters, params=params)
    data_series = data_msg.to_pandas()
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({"Unemployment Rate": data_series})
    else:
        try:
            df = pd.DataFrame({"Unemployment Rate": data_series.xs(
                (sex_code, age_code, country_code, adjust_code, "PC_ACT"),
                level=[2, 3, 4, 1, 0]
            )})
        except:
            df = pd.DataFrame(data_series)
            df.columns = ["Unemployment Rate"]
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df = df.sort_index()
    return df

# تابع واکشی داده کمکی Eurostat
@st.cache_data(show_spinner=False)
def fetch_aux_from_eurostat(dataset_code, country_code, sex_code, age_code):
    estat_req = Request('ESTAT')
    key_filters = {'sex': sex_code, 'age': age_code, 'geo': country_code}
    if dataset_code == 'lfsq_agan':
        key_filters['unit'] = 'THS'
    elif dataset_code == 'lfsq_argan':
        key_filters['unit'] = 'PC_POP'
    try:
        data_msg = estat_req.data(dataset_code, key=key_filters)
    except Exception:
        key_filters.pop('unit', None)
        data_msg = estat_req.data(dataset_code, key=key_filters)
    data_series = data_msg.to_pandas()
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({dataset_code: data_series})
    else:
        try:
            df = pd.DataFrame(data_series.xs((sex_code, age_code, country_code), level=[1, 2, 3]))
            df.columns = [dataset_code]
        except:
            df = pd.DataFrame(data_series)
            df.columns = [dataset_code]
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df = df.sort_index()
    return df

# تابع واکشی داده کمکی ISTAT
@st.cache_data(show_spinner=False)
def fetch_aux_from_istat(flow_id, country_code, sex_code, age_code, adjust_code):
    freq = "M"
    region = country_code
    # نگاشت flow_id به data_type (در صورت وجود)
    if flow_id == "150_876":
        data_type = "ACTIVITY_RATE"  # فرضی
    elif flow_id == "150_873":
        data_type = "LABOUR_FORCE"   # فرضی
    else:
        data_type = None
    ed_code = get_latest_edition_code(get_istat_editions())
    if ed_code is None:
        raise ValueError("Edition داده‌های ISTAT نامشخص است")
    if data_type:
        sdmx_key = f"{freq}.{region}.{data_type}.{adjust_code}.{sex_code}.{age_code}.{ed_code}"
    else:
        sdmx_key = f"{freq}.{region}.{adjust_code}.{sex_code}.{age_code}.{ed_code}"
    url = f"https://esploradati.istat.it/SDMXWS/rest/data/ISTAT/{flow_id}/{sdmx_key}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    istat_req = Request('ISTAT')
    key_filters = {'FREQ': freq, 'REF_AREA': region, 'SEX': sex_code, 'AGE': age_code, 'ADJUSTMENT': adjust_code}
    if data_type:
        key_filters['DATA_TYPE'] = data_type
    data_msg = istat_req.data(flow_id, key=key_filters)
    data_series = data_msg.to_pandas()
    if isinstance(data_series, pd.Series):
        df = pd.DataFrame({flow_id: data_series})
    else:
        df = pd.DataFrame(data_series)
        df.columns = [flow_id]
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df = df.sort_index()
    return df

# --- رابط کاربری ---
st.set_page_config(page_title="نرخ بیکاری - SDMX", layout="wide")
st.title("نمایش نرخ بیکاری ماهانه")

# ویجت‌های کناری:
countries = {"ایتالیا": "IT", "آلمان": "DE", "فرانسه": "FR", "اسپانیا": "ES", "اتحادیه اروپا": "EU27_2020"}
country_name = st.sidebar.selectbox("کشور / منطقه:", list(countries.keys()), index=0)
country_code = countries[country_name]

sex_options = {"کل جمعیت": "9", "مردان": "1", "زنان": "2"}  # ISTAT codes
sex_label = st.sidebar.selectbox("جنسیت:", list(sex_options.keys()), index=0)
sex_code_istat = sex_options[sex_label]
sex_code_eurostat = "T" if sex_code_istat == "9" else ("M" if sex_code_istat == "1" else "F")

age_options = {
    "15-74 (کل)": "Y15-74",
    "15-24 (جوانان)": "Y15-24",
    "25-74 (بزرگسالان)": "Y25-74",
    "15-34": "Y15-34", "35-49": "Y35-49",
    "50-64": "Y50-64", "50-74": "Y50-74",
    "15-64": "Y15-64"
}
age_label = st.sidebar.selectbox("گروه سنی:", list(age_options.keys()), index=0)
age_code_istat = age_options[age_label]
# نگاشت گروه سنی به Eurostat
if age_code_istat == "Y15-74":
    age_code_eurostat = "TOTAL"
elif age_code_istat in ["Y15-24", "Y15-34"]:
    age_code_eurostat = "Y_LT25"
elif age_code_istat == "Y25-74":
    age_code_eurostat = "Y25-74"
else:
    age_code_eurostat = "TOTAL"  # سایر موارد روی کل تخمین زده می‌شوند

adjust_options = {"خام (NSA)": "NSA", "تعدیل فصلی (SA)": "SA", "روند (Trend)": "TC"}
adjust_label = st.sidebar.selectbox("تعدیل:", list(adjust_options.keys()), index=1)
adjust_code_eurostat = adjust_options[adjust_label]
# نگاشت به ISTAT
adjust_code_istat = "N" if adjust_code_eurostat == "NSA" else ("Y" if adjust_code_eurostat == "SA" else "N")

start_year = st.sidebar.number_input("سال شروع:", min_value=2000, max_value=2030, value=2015)
start_month = st.sidebar.selectbox("ماه شروع:", list(range(1, 13)), index=0)
end_year = st.sidebar.number_input("سال پایان:", min_value=2000, max_value=2030, value=2025)
end_month = st.sidebar.selectbox("ماه پایان:", list(range(1, 13)), index=9)
start_period = f"{start_year}-{start_month:02d}"
end_period = f"{end_year}-{end_month:02d}"

aux_participation = st.sidebar.checkbox("نرخ مشارکت اقتصادی (نرخ فعالیت)")
aux_active_pop = st.sidebar.checkbox("جمعیت فعال (هزار نفر)")

if st.sidebar.button("نمایش نتایج"):
    data_source = None
    main_df = None
    aux_dfs = {}
    use_istat = (country_code == "IT")
    # اگر TrendCycle برای ایتالیا انتخاب شده باشد، از Eurostat استفاده می‌کنیم چون ISTAT ارائه نمی‌کند
    if adjust_code_eurostat == "TC" and country_code == "IT":
        st.info("مولفه روند از Eurostat واکشی خواهد شد (در ISTAT موجود نیست).")
        use_istat = False
    if use_istat:
        try:
            main_df = fetch_from_istat(country_code, sex_code_istat, age_code_istat, adjust_code_istat, start_period, end_period)
            data_source = "ISTAT"
        except Exception:
            use_istat = False
    if not use_istat:
        try:
            main_df = fetch_from_eurostat(country_code, sex_code_eurostat, age_code_eurostat, adjust_code_eurostat, start_period, end_period)
            data_source = "Eurostat"
        except Exception:
            data_source = "sample"
            date_idx = pd.date_range(start=pd.to_datetime(start_period), end=pd.to_datetime(end_period), freq='M')
            sample_values = np.linspace(10, 6, num=len(date_idx)) + np.random.randn(len(date_idx)) * 0.3
            main_df = pd.DataFrame({"Unemployment Rate": sample_values}, index=date_idx)
            st.warning("عدم امکان واکشی داده‌های واقعی - نمایش داده نمونه")
    if main_df is not None:
        # واکشی داده‌های کمکی در صورت انتخاب
        if aux_participation:
            if data_source == "ISTAT" and country_code == "IT":
                try:
                    aux_df = fetch_aux_from_istat("150_876", country_code, sex_code_istat, age_code_istat, adjust_code_istat)
                    aux_dfs["Activity Rate"] = aux_df
                except Exception:
                    try:
                        aux_df = fetch_aux_from_eurostat("lfsq_argan", country_code, sex_code_eurostat, age_code_eurostat)
                        aux_dfs["Activity Rate"] = aux_df
                    except:
                        st.error("خطا در واکشی نرخ مشارکت")
            else:
                try:
                    aux_df = fetch_aux_from_eurostat("lfsq_argan", country_code, sex_code_eurostat, age_code_eurostat)
                    aux_dfs["Activity Rate"] = aux_df
                except:
                    st.error("خطا در واکشی نرخ مشارکت")
        if aux_active_pop:
            if data_source == "ISTAT" and country_code == "IT":
                try:
                    aux_df = fetch_aux_from_istat("150_873", country_code, sex_code_istat, age_code_istat, adjust_code_istat)
                    aux_dfs["Active Population"] = aux_df
                except Exception:
                    try:
                        aux_df = fetch_aux_from_eurostat("lfsq_agan", country_code, sex_code_eurostat, age_code_eurostat)
                        aux_dfs["Active Population"] = aux_df
                    except:
                        st.error("خطا در واکشی جمعیت فعال")
            else:
                try:
                    aux_df = fetch_aux_from_eurostat("lfsq_agan", country_code, sex_code_eurostat, age_code_eurostat)
                    aux_dfs["Active Population"] = aux_df
                except:
                    st.error("خطا در واکشی جمعیت فعال")
        if data_source == "Eurostat" and country_code == "IT":
            st.info("داده‌ها از Eurostat نمایش یافته‌اند (fallback از ISTAT).")
        # رسم نمودار
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(main_df.index, main_df["Unemployment Rate"], label="نرخ بیکاری", color='C0', linewidth=2)
        ax.set_title(f"نرخ بیکاری ماهانه - {country_name} ({sex_label}, {age_label})")
        ax.set_xlabel("تاریخ")
        ax.set_ylabel("نرخ بیکاری (%)")
        # آماده‌سازی برای رسم داده‌های کمکی
        aux_percent = {}   # متغیرهای کمکی درصدی (نرخ‌ها)
        aux_absolute = {}  # متغیرهای کمکی مقداری (جمعیت)
        for name, df_aux in aux_dfs.items():
            if name == "Active Population":
                aux_absolute[name] = df_aux
            else:
                aux_percent[name] = df_aux
        # رسم متغیرهای درصدی بر روی محور اصلی
        for name, df_aux in aux_percent.items():
            df_res = df_aux.copy()
            # در صورت فصلی بودن، به ماهانه تبدیل می‌کنیم
            if df_res.index.inferred_freq is None or df_res.index.freq != 'M':
                df_res = df_res.resample('M').ffill()
            ax.plot(df_res.index, df_res[df_res.columns[0]], label=name, linestyle='--')
        # اگر متغیر مقداری (مثل جمعیت فعال) داریم، محور دوم اضافه می‌کنیم
        ax2 = None
        if aux_absolute:
            ax2 = ax.twinx()
            for name, df_aux in aux_absolute.items():
                df_res = df_aux.copy()
                if df_res.index.inferred_freq is None or df_res.index.freq != 'M':
                    df_res = df_res.resample('M').ffill()
                ax2.plot(df_res.index, df_res[df_res.columns[0]], label=name, color='green', linestyle=':')
            ax2.set_ylabel("جمعیت فعال (هزار نفر)")
        # ترکیب legend‌ها از دو محور (در صورت وجود محور دوم)
        lines1, labels1 = ax.get_legend_handles_labels()
        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
        st.pyplot(fig)
        # محاسبه شاخص‌های آماری
        latest_val = main_df["Unemployment Rate"].iloc[-1]
        avg_val = main_df["Unemployment Rate"].mean()
        change_mom = None
        change_yoy = None
        if len(main_df) >= 2:
            change_mom = latest_val - main_df["Unemployment Rate"].iloc[-2]
        if len(main_df) > 12:
            change_yoy = latest_val - main_df["Unemployment Rate"].iloc[-13]
        st.subheader("شاخص‌های کلیدی:")
        cols = st.columns(4)
        cols[0].metric("آخرین مقدار", f"{latest_val:.2f}%")
        cols[1].metric("میانگین", f"{avg_val:.2f}%")
        cols[2].metric("تغییر ماهانه", f"{change_mom:+.2f} واحد" if change_mom is not None else "نمیر")
        cols[3].metric("تغییر سالانه", f"{change_yoy:+.2f} واحد" if change_yoy is not None else "نمیر")
        # محاسبه و نمایش همبستگی با متغیرهای کمکی
        if aux_dfs:
            st.subheader("همبستگی با متغیرهای انتخاب‌شده:")
            for name, df_aux in aux_dfs.items():
                combined = pd.DataFrame({"unemp": main_df["Unemployment Rate"]})
                df_res = df_aux.copy()
                if df_res.index.inferred_freq is None or df_res.index.freq != 'M':
                    df_res = df_res.resample('M').ffill()
                combined[name] = df_res[df_res.columns[0]]
                combined = combined.dropna()
                if combined.shape[0] < 2:
                    st.write(f"محاسبه همبستگی برای {name}: داده کافی نیست.")
                else:
                    corr_val = combined["unemp"].corr(combined[name])
                    st.write(f"ضریب همبستگی بین نرخ بیکاری و **{name}**: {corr_val:.2f}")
