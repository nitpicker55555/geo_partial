import shapely
import geopandas as gpd
import r5py
import json
from datetime import datetime, timedelta


# 载入交通网络
transport_network = r5py.TransportNetwork(
    "raw_file/Muenchen.osm.pbf",
    ["raw_file/stop_times.zip"]
)


def extract_mode_and_number(s: str):
    s=str(s)
    # print(s)
    if 'TransportMode' in s:
        parts = s.split()
        if len(parts) < 2:
            return None, None  # 确保有足够的部分

        # 提取交通方式（去掉TransportMode.前缀）
        mode = parts[0].split('.')[-1]

        # 提取编号部分
        number_parts = parts[1].split('-')
        if len(number_parts) < 2:
            return mode, None  # 确保有足够的部分
        return f"{mode}-{number_parts[1]}"
    else:
        return s

def compute_multi_stop_route( waypoints):
    departure_time = datetime(2025, 3, 10, 8, 0, 0)
    max_time = timedelta(minutes=60)

    total_travel_time = 0
    full_itinerary = []
    itinerary_json = {}
    for i in range(len(waypoints) - 1):
        start_coordinates = waypoints[i]
        end_coordinates = waypoints[i + 1]

        origins = gpd.GeoDataFrame(
            {"id": ["origin"],
             "geometry": [shapely.geometry.Point(start_coordinates[1], start_coordinates[0])]},
            crs="EPSG:4326"
        )

        destinations = gpd.GeoDataFrame(
            {"id": ["destination"],
             "geometry": [shapely.geometry.Point(end_coordinates[1], end_coordinates[0])]},
            crs="EPSG:4326"
        )

        detailed_itineraries = r5py.DetailedItinerariesComputer(
            transport_network,
            origins=origins,
            destinations=destinations,
            departure=departure_time,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=max_time
        )

        itineraries = detailed_itineraries.compute_travel_details()
        options = {}

        for index, row in itineraries.iterrows():
            # print(row.keys())
            option_id = row["option"]
            if option_id not in options:
                options[option_id] = {"segments": [], "total_travel_time": 0}

            travel_time = int(row["travel_time"].total_seconds() / 60)
            options[option_id]["segments"].append({
                "transport_mode": row["transport_mode"],
                "travel_time": travel_time,
                "route": row.get("route", None),
                "geometry": row["geometry"],
                "route_id": row["route_id"],
                'start_stop_id': row["start_stop_id"],
                'end_stop_id': row["end_stop_id"],
                'wait_time': row["wait_time"],
                "start_wp": i + 1,
                "end_wp": i + 2
            })
            options[option_id]["total_travel_time"] += travel_time

        earliest_option = min(options.items(), key=lambda x: x[1]["total_travel_time"], default=None)

        if earliest_option:
            option_id, data = earliest_option
            total_travel_time += data["total_travel_time"]
            departure_time += timedelta(minutes=data["total_travel_time"])
            full_itinerary.extend(data["segments"])
        else:
            print(f"无法找到从 {start_coordinates} 到 {end_coordinates} 的路线。")
            return None
    if full_itinerary:
        print(f"总旅行时间: {total_travel_time} 分钟")
        itinerary_json = {"total_travel_time": total_travel_time, "segments": {}}

        for i, segment in enumerate(full_itinerary, 1):

            # print(f"  Segment {i} (Waypoints {segment['start_wp']} → {segment['end_wp']}):")
            # print(f"    交通方式: {segment['transport_mode']}, 线路: {segment['route_id']}")
            # print(f"    start_stop_id: {segment['start_stop_id']}, end_stop_id: {segment['end_stop_id']}")
            # print(f"    wait_time: {segment['wait_time']}")
            # print(f"    旅行时间: {segment['travel_time']} 分钟")
            # print(f"    路径: {segment['geometry']}")
            if segment['route_id']:
                segment['transport_mode'] = extract_mode_and_number(str(segment['transport_mode']) + " " + str(segment['route_id']))
            else:
                segment['transport_mode'] = "WALK"
            # itinerary_json["segments"].update({
            #     "segment_name": f"From {segment['start_wp']} To {segment['end_wp']}",
            #     "transport_mode": str(segment['transport_mode']),
            #     "travel_time": segment['travel_time'],
            #     "geometry": segment['geometry']
            # })
            itinerary_json["segments"].update({
                # "segment_name": f"From {segment['start_wp']} To {segment['end_wp']}",
                # "transport_mode": str(segment['transport_mode']),
                # "travel_time": segment['travel_time'],
                f"Path {segment['start_wp']}_{str(segment['transport_mode'])}_TIME:{str(segment['travel_time'])}_{str(i)}": segment['geometry']
            })
        # print(itinerary_json)

    return total_travel_time, full_itinerary, itinerary_json

#
# # 示例输入
# waypoints = [
#
#     (48.149636, 11.5669883),  # 起点
#     (48.14871, 11.5749857),  # 中间站点
#     (48.1836522, 11.6057038),  # 终点
#
# ]
#
#
#
# # 计算路线
# total_travel_time, itinerary,result = compute_multi_stop_route(waypoints, departure_time, max_time)
# print(result)
# 输出结果
