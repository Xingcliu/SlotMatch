from pathlib import Path
import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from spatial_metrics import distance_calculator
from delivery_environment import PureRealRoadNetworkEnvironmentWithConstraints, RealRoadOrder, VehicleState

def peek_orders(env):
    combined = dict(env.active_orders)
    for payload in env.orders_data.get(env.time_step, {}).values():
        if not isinstance(payload, list) or not payload:
            continue
        row = payload[0] if isinstance(payload[0], list) else payload
        if len(row) < 5:
            continue
        (oid, start, ddl, src, dst) = map(int, row[:5])
        if src not in env.road_graph or dst not in env.road_graph:
            continue
        if env.node_component.get(src) != env.node_component.get(dst):
            continue
        combined[oid] = RealRoadOrder(oid, src, dst, start, ddl, 0.5, env.time_step)
    return sorted(list(combined.values())[:env.max_concurrent_orders], key=lambda o: o.order_id)

def oracle_action(env, uav_bonus=40.0, uav_only=False, fast_carrier_candidates=False):
    orders = peek_orders(env)
    uavs = sorted([v for v in env.vehicles if v.vehicle_type == 'uav' and env.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE], key=lambda v: v.vehicle_id)
    carriers = [] if uav_only else sorted([v for v in env.vehicles if v.vehicle_type == 'carrier' and env.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE], key=lambda v: v.vehicle_id)
    vehicles = uavs + carriers
    dispatch = np.zeros(15, np.float32)
    uw = np.zeros((env.num_uavs, 15), np.float32)
    cw = np.zeros((env.num_ground_vehicles, 15), np.float32)
    if not orders or not vehicles:
        return (dispatch, uw.ravel(), cw.ravel())
    score = np.full((len(vehicles), len(orders)), -1000000000.0)
    for (vi, vehicle) in enumerate(vehicles):
        is_uav = vi < len(uavs)
        for (oi, order) in enumerate(orders):
            if not (fast_carrier_candidates and (not is_uav)):
                if not env._can_vehicle_handle_order(vehicle, order):
                    continue
            if is_uav:
                d1 = distance_calculator.get_straight_line_distance(vehicle.current_node, order.start_node)
                d2 = distance_calculator.get_straight_line_distance(order.start_node, order.end_node)
                score[vi, oi] = 1000 + uav_bonus - 10 * d1 - 2 * d2
            else:
                if fast_carrier_candidates:
                    d1 = distance_calculator.get_straight_line_distance(vehicle.current_node, order.start_node)
                    d2 = distance_calculator.get_straight_line_distance(order.start_node, order.end_node)
                else:
                    d1 = distance_calculator.get_road_network_distance(vehicle.current_node, order.start_node)
                    d2 = distance_calculator.get_road_network_distance(order.start_node, order.end_node)
                score[vi, oi] = 1000 - 10 * d1 - 2 * d2
    side = len(vehicles) + len(orders)
    aug = np.zeros((side, side))
    aug[:len(vehicles), :len(orders)] = score
    (rows, cols) = linear_sum_assignment(aug, maximize=True)
    for (vi, oi) in zip(rows, cols):
        if vi >= len(vehicles) or oi >= len(orders) or score[vi, oi] <= 0:
            continue
        if vi < len(uavs):
            dispatch[oi] = 1
            uw[vi, oi] = 1
        else:
            cw[vi - len(uavs), oi] = 1
    return (dispatch, uw.ravel(), cw.ravel())
