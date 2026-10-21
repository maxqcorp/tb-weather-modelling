"""Origin-restricted feature construction from local aggregate source tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE = ['tb_count_lag1','tb_count_lag2','tb_count_lag4','tb_count_lag8',
        'tb_count_prev4w_mean','annual_sin','annual_cos','week_index','covid_stringency_prevweek']
CORE = ['temp_mean_c','rh_mean_pct','rain_total_mm']
EXTRA = ['wind_mean_ms','solar_total_mjm2','temp_range_c']
WEATHER = CORE+EXTRA
FEATURES = {'B':BASE,'C':BASE+[f'era5_{x}_prev4w' for x in CORE],
            'E':BASE+[f'era5_{x}_prev4w' for x in WEATHER]}
DISTRICTS=['CHERAS','KEPONG','LEMBAH PANTAI','TITIWANGSA','WP PUTRAJAYA']
CITIES=['Kuala Lumpur','Putrajaya']
CITY_MAP={d:('Putrajaya' if d=='WP PUTRAJAYA' else 'Kuala Lumpur') for d in DISTRICTS}


class DataStore:
    def __init__(self, data_root):
        self.root=Path(data_root)
        district=self.root/'climate/district_grid_2019_2021'
        city=self.root/'climate/kl_putrajaya_2019_2021'
        self.paths={'district_weekly':district/'district_weekly_tb_weather.csv',
                    'city_weekly':city/'weekly_tb_weather_2020_2021.csv',
                    'city_daily':city/'daily_weather_all_sources.csv',
                    'policy':city/'malaysia_covid_stringency_2020_2021.csv',
                    'footprint_daily':district/'district_weather_proxies_daily.csv',
                    'footprint_weights':district/'district_proxy_grid_weights.csv'}
        self.weekly=pd.read_csv(self.paths['district_weekly'],parse_dates=['week_start','week_end'])
        self.city_weekly=pd.read_csv(self.paths['city_weekly'],parse_dates=['week_start','week_end'])
        self.daily=pd.read_csv(self.paths['city_daily'],parse_dates=['date'])
        self.policy=pd.read_csv(self.paths['policy'],parse_dates=['date'])
        self.proxy=pd.read_csv(self.paths['footprint_daily'],parse_dates=['date'])
        assert not self.weekly.duplicated(['district','week_start']).any()
        assert not self.daily.duplicated(['city','date']).any()
        assert not self.policy.date.duplicated().any()

    def counts(self, geography='district', revised=False):
        column='target_notifications_excl_revised' if revised else 'target_notifications'
        panel=self.weekly[['district','week_start','week_end','full_study_week',column]].copy()
        panel=panel.rename(columns={column:'target'})
        if geography=='city':
            panel['unit']=panel.district.map(CITY_MAP)
            panel=panel.groupby(['unit','week_start','week_end','full_study_week'],as_index=False).target.sum()
        else:
            panel=panel.rename(columns={'district':'unit'})
        panel=panel.sort_values(['week_start','unit']).reset_index(drop=True)
        panel['full_study_week']=(panel.week_start>=pd.Timestamp('2020-01-01')) & (panel.week_end<=pd.Timestamp('2021-12-31'))
        return panel

    def daily_weather(self, unit, geography='district', source='era5'):
        if source=='footprint':
            d=self.proxy.loc[self.proxy.district.eq(unit),['date']+WEATHER].copy()
        else:
            city=unit if geography=='city' else CITY_MAP[unit]
            d=self.daily.loc[self.daily.city.eq(city),['date']].copy()
            mask=self.daily.city.eq(city)
            for name in WEATHER:
                provider='era5land' if source=='mixed' and name in ['temp_mean_c','rh_mean_pct','temp_range_c'] else ('era5' if source=='mixed' else source)
                d[name]=self.daily.loc[mask,f'{provider}_{name}'].to_numpy()
        return d.set_index('date').sort_index()


def origin_features(panel, weather, policy, unit, origin, window=28, delay=0, omit_policy=False):
    """No target or future records enter X; missing structural history returns None."""
    t=pd.Timestamp(origin)
    p=panel.loc[panel.unit.eq(unit)&panel.week_end.lt(t)].set_index('week_start')
    x={}
    for k in [1,2,4,8]:
        key=t-pd.Timedelta(weeks=k+delay)
        if key not in p.index or not bool(p.at[key,'full_study_week']):return None
        x[f'tb_count_lag{k}']=float(p.at[key,'target'])
    prev=pd.date_range(t-pd.Timedelta(weeks=4+delay),periods=4,freq='W-MON')
    if not prev.isin(p.index).all() or not p.loc[prev,'full_study_week'].all():return None
    x['tb_count_prev4w_mean']=float(p.loc[prev,'target'].mean())
    x['annual_sin']=float(np.sin(2*np.pi*t.dayofyear/365.25))
    x['annual_cos']=float(np.cos(2*np.pi*t.dayofyear/365.25))
    x['week_index']=int((t-pd.Timestamp('2019-12-30')).days//7)
    end=t-pd.Timedelta(weeks=delay)
    pol=policy.loc[policy.date.ge(end-pd.Timedelta(days=7))&policy.date.lt(end)]
    if len(pol)!=7 or pol.covid_stringency_index.isna().any():return None
    if not omit_policy:x['covid_stringency_prevweek']=float(pol.covid_stringency_index.mean())
    days=pd.date_range(end-pd.Timedelta(days=window),periods=window,freq='D')
    if not days.isin(weather.index).all():return None
    w=weather.loc[days,WEATHER]
    if w.isna().any().any():return None
    assert w.index.max()<t
    for name in WEATHER:
        x[f'era5_{name}_prev4w']=float(w[name].sum() if name in ['rain_total_mm','solar_total_mjm2'] else w[name].mean())
    assert all(np.isfinite(v) for v in x.values())
    return x


def build_dataset(store, geography='district', revised=False, source='era5',window=28,delay=0,omit_policy=False):
    panel=store.counts(geography,revised)
    units=DISTRICTS if geography=='district' else CITIES
    rows=[]
    for unit in units:
        weather=store.daily_weather(unit,geography,source)
        for row in panel.loc[panel.unit.eq(unit)&panel.full_study_week].itertuples():
            x=origin_features(panel,weather,store.policy,unit,row.week_start,window,delay,omit_policy)
            if x is None:continue
            partition='train' if row.week_end.year==2020 else ('test' if row.week_start.year==2021 else 'bridge')
            rows.append({'unit':unit,'week_start':row.week_start,'week_end':row.week_end,'partition':partition,'target':int(row.target),**x})
    f=pd.DataFrame(rows).sort_values(['week_start','unit']).reset_index(drop=True)
    assert not f.duplicated(['unit','week_start']).any()
    assert f.groupby('week_start').unit.nunique().eq(len(units)).all()
    return f,units


def assert_primary_matches_prepared(data, prepared_path):
    import json
    old=pd.DataFrame(json.loads(Path(prepared_path).read_text())['rows'])
    old.week_start=pd.to_datetime(old.week_start)
    merged=data.merge(old,left_on=['unit','week_start'],right_on=['district','week_start'],validate='one_to_one',suffixes=('','_prepared'))
    assert len(data)==len(merged)==475
    assert np.array_equal(merged.target,merged.target_notifications)
    for name in FEATURES['E']:
        assert np.allclose(merged[name],merged[name+'_prepared'],rtol=0,atol=1e-5),name
    assert data.partition.value_counts().to_dict()=={'test':255,'train':215,'bridge':5}
