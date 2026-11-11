import numpy as np
import networkx as nx
import torch
import pickle
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from fleet_dynamics import RoadNetworkUAV, RoadNetworkCarrier, RoadNetworkOrder
from spatial_metrics import distance_calculator
from route_store import cached_shortest_path_length, cached_shortest_path, print_global_cache_stats, clear_global_cache
from enum import Enum

class VehicleState(Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    DELIVERING = "delivering"
    FLYING_TO_CHARGE = "flying_to_charge"

class OrderStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    COMPLETED = "completed"
    EXPIRED = "expired"

@dataclass
class RealRoadOrder:
    order_id: int
    start_node: int
    end_node: int
    start_time: int
    deadline: int
    weight: float
    created_time: int

class PureRealRoadNetworkEnvironmentWithConstraints:
    
    def _get_global_order_distribution(self):
        if not self.active_orders:
            return np.zeros(8, dtype=np.float32)
        
        total_orders = len(self.active_orders)
        urgent_orders = sum(1 for order in self.active_orders.values() if self._is_order_urgent(order))
        
        order_nodes = [order.start_node for order in self.active_orders.values()]
        avg_order_node = np.mean(order_nodes) if order_nodes else 0
        
        matched_orders = sum(1 for vehicle in self.vehicles if vehicle.vehicle_id in self.vehicle_assigned_orders)
        
        global_info = np.array([
            total_orders / 50.0,
            urgent_orders / max(total_orders, 1),
            matched_orders / self.num_vehicles,
            self.time_step / self.max_time_steps,
            len(self.completed_orders) / max(self.total_orders_generated, 1),
            self.total_orders_matched / max(self.total_orders_generated, 1),
            avg_order_node / 600000.0,
            min(len(self.active_orders) / 15.0, 1.0)
        ], dtype=np.float32)
        
        return global_info

    def __init__(self, num_ground_vehicles, num_uavs, max_time_steps, max_concurrent_orders=15):
        self.num_ground_vehicles = num_ground_vehicles
        self.num_uavs = num_uavs
        self.num_vehicles = num_ground_vehicles + num_uavs
        self.max_concurrent_orders = max_concurrent_orders
        
        self.road_graph = None
        self.node_features = None
        self.node_embeddings = {}
        self.largest_component_nodes = None
        
        self.vehicles: List = []
        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        self.failed_orders = []
        self.active_orders: Dict[int, RealRoadOrder] = {}
        self.completed_orders: List[RealRoadOrder] = []
        
        self.time_step = 0
        self.max_time_steps = max_time_steps
        self.orders_data = None
        
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0
        self.uav_energy_failures = 0
        self.charging_events = 0
        self.episode_count = 0
        
        self.episode_pickups = 0
        self.episode_deliveries = 0
        self.episode_uav_deliveries = 0
        self.episode_carrier_deliveries = 0
        self.total_uav_deliveries = 0
        self.total_carrier_deliveries = 0
        
        self._load_real_road_network()
        self._initialize_vehicles_with_constraints()
        
    def _load_grid_mapping_nodes(self):
        try:
            import pandas as pd
            mapping_file = "./shanghai_road_dataset/mapping/grid_to_node_mapping.csv"
            df = pd.read_csv(mapping_file)
            node_ids = df.iloc[:, 1].values.astype(int)
            self.mapping_node_ids = np.unique(node_ids)
            return True
        except Exception as e:
            self.mapping_node_ids = None
            return False

    def _load_real_road_network(self):
        try:
            with open('./shanghai_road_dataset/road_network/road_graph_fixed.pkl', 'rb') as f:
                self.road_graph = pickle.load(f)
            distance_calculator.set_road_graph(self.road_graph)
            
            self.node_features = np.load('./shanghai_road_dataset/road_network/node_features_fixed.npy')
            
            self.node_embeddings = {}
            for node_id in self.road_graph.nodes:
                if node_id < len(self.node_features):
                    raw_features = self.node_features[node_id][:8]
                    normalized_features = raw_features.copy()
                    normalized_features[0] = (raw_features[0] - 121.0) / 1.0
                    normalized_features[1] = (raw_features[1] - 31.0) / 1.0
                    normalized_features[2:] = raw_features[2:] / (np.abs(raw_features[2:]) + 1e-6)
                    
                    self.node_embeddings[node_id] = normalized_features.astype(np.float32)
                else:
                    self.node_embeddings[node_id] = np.zeros(8, dtype=np.float32)
            
            connected_components = list(nx.connected_components(self.road_graph))
            self.largest_component_nodes = list(max(connected_components, key=len))
            self.node_component = {
                node: component_id
                for component_id, component in enumerate(connected_components)
                for node in component
            }
            
            self._load_grid_mapping_nodes()
            
        except Exception as e:
            raise RuntimeError("Road network data loading failed")
    
    def _initialize_vehicles_with_constraints(self):
        self.vehicles = []
        
        if hasattr(self, "mapping_node_ids") and self.mapping_node_ids is not None and self.road_graph is not None:
            valid_mapping_nodes = [node for node in self.mapping_node_ids if node in self.road_graph.nodes()]
            if len(valid_mapping_nodes) >= self.num_vehicles:
                available_nodes = np.random.choice(valid_mapping_nodes, self.num_vehicles, replace=False)
            else:
                available_nodes = np.random.choice(valid_mapping_nodes, self.num_vehicles, replace=True)
        else:
            if self.largest_component_nodes is None or len(self.largest_component_nodes) < self.num_vehicles:
                if self.largest_component_nodes is None:
                    available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
                else:
                    available_nodes = self.largest_component_nodes
            else:
                available_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
        
        num_charging_stations = 15
        if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= num_charging_stations:
            charging_stations = np.random.choice(self.largest_component_nodes, num_charging_stations, replace=False)
        else:
            charging_stations = self.largest_component_nodes[:num_charging_stations] if self.largest_component_nodes is not None else [0] * num_charging_stations
        
        self.charging_stations = charging_stations
        
        for i in range(self.num_ground_vehicles):
            start_node = available_nodes[i % len(available_nodes)]
            carrier = RoadNetworkCarrier(
                carrier_id=i, 
                start_node=start_node, 
                road_graph=self.road_graph,
                capacity=float('inf'),
                speed=90
            )
            self.vehicles.append(carrier)
            self.vehicle_states[carrier.vehicle_id] = VehicleState.IDLE
        
        for i in range(self.num_uavs):
            vehicle_id = self.num_ground_vehicles + i
            start_node = available_nodes[vehicle_id % len(available_nodes)]
            charging_station = charging_stations[i % len(charging_stations)]
            
            uav = RoadNetworkUAV(
                uav_id=vehicle_id,
                start_node=start_node,
                road_graph=self.road_graph,
                capacity=5,
                battery_capacity=1,
                charging_station_node=charging_station,
                charging_stations_list=charging_stations,
                speed=120
            )
            self.vehicles.append(uav)
            self.vehicle_states[uav.vehicle_id] = VehicleState.IDLE
    
    def load_road_orders(self, day: int) -> bool:
        try:
            order_file = (
                f'./shanghai_road_dataset/processed_orders_road/'
                f'Orders_Dataset_shanghai_road_day_{day}'
            )
            with open(order_file, 'rb') as f:
                self.orders_data = pickle.load(f)
            
            return True
            
        except Exception as e:
            return False
    
    def generate_orders(self) -> List[RealRoadOrder]:
        new_orders = []
        
        if not hasattr(self, 'orders_data') or not self.orders_data or self.time_step not in self.orders_data:
            return new_orders
        
        if self.road_graph is None:
            return new_orders
        
        time_step_orders = self.orders_data[self.time_step]
        
        for order_key, order_data in time_step_orders.items():
            try:
                if isinstance(order_data, list) and len(order_data) > 0:
                    if isinstance(order_data[0], list):
                        actual_data = order_data[0]
                    else:
                        actual_data = order_data
                else:
                    continue
                
                if len(actual_data) >= 5:
                    road_order_id, start_time, end_time, start_node, end_node = actual_data[:5]
                    
                    if (self.road_graph is not None and 
                        start_node in self.road_graph.nodes and 
                        end_node in self.road_graph.nodes):
                        # Exact equivalent of the public unweighted has-path
                        # filter on this undirected graph, implemented as O(1)
                        # component lookup to avoid repeated whole-graph BFS.
                        if self.node_component.get(start_node) != self.node_component.get(end_node):
                            continue
                        
                        order = RealRoadOrder(
                            order_id=int(road_order_id),
                            start_node=int(start_node),
                            end_node=int(end_node),
                            start_time=int(start_time),
                            deadline=int(end_time),
                            weight=0.5,
                            created_time=int(self.time_step)
                        )
                        
                        new_orders.append(order)
                        self.active_orders[order.order_id] = order
            
            except Exception as e:
                continue
        
        self.total_orders_generated += len(new_orders)
        
        return new_orders

    def assign_order_to_vehicle(self, vehicle_id, order):
        if self.vehicle_states.get(vehicle_id) != VehicleState.IDLE:
            return False
        
        self.vehicle_states[vehicle_id] = VehicleState.ASSIGNED
        self.vehicle_assigned_orders[vehicle_id] = order
        self.order_statuses[order.order_id] = OrderStatus.ASSIGNED
        
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]

        return True
    
    def vehicle_start_pickup(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            self.order_statuses[order.order_id] = OrderStatus.PICKED_UP
            
            self.episode_pickups += 1
            return True
        return False
    
    def vehicle_start_delivery(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            self.vehicle_states[vehicle_id] = VehicleState.DELIVERING
            return True
        return False
    
    def vehicle_complete_delivery(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.DELIVERING and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            
            self.order_statuses[order.order_id] = OrderStatus.COMPLETED
            
            self.completed_orders.append(order)
            self.total_orders_completed += 1
            self.episode_deliveries += 1
            
            del self.vehicle_assigned_orders[vehicle_id]
            
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)

            if vehicle and hasattr(vehicle, 'vehicle_type'):
                if vehicle.vehicle_type == 'uav':
                    self.episode_uav_deliveries += 1
                    self.total_uav_deliveries += 1
                elif vehicle.vehicle_type == 'carrier':
                    self.episode_carrier_deliveries += 1
                    self.total_carrier_deliveries += 1
            
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                need_charge, reason = vehicle.needs_return_to_charge()
                if need_charge:
                    charge_result = vehicle.return_to_charging_station()
                    
                    if charge_result == True:
                        self.vehicle_states[vehicle_id] = VehicleState.IDLE
                        self.charging_events += 1
                    elif charge_result == "flying_to_charge":
                        self.vehicle_states[vehicle_id] = VehicleState.FLYING_TO_CHARGE
                    else:
                        self.vehicle_states[vehicle_id] = VehicleState.IDLE
                else:
                    self.vehicle_states[vehicle_id] = VehicleState.IDLE
            else:
                self.vehicle_states[vehicle_id] = VehicleState.IDLE
            
            return True
        return False
        
    def _can_complete_order_within_ddl(self, vehicle, order):
        try:
            if not self._ddl_prefilter_by_straight_line(vehicle, order):
                return False
            
            if vehicle.vehicle_type == 'uav':
                pickup_time = vehicle.calculate_flight_time_to_node(order.start_node)
                delivery_time = vehicle.calculate_flight_time_to_node(order.end_node)
            else:
                pickup_time = vehicle.calculate_road_travel_time(vehicle.current_node, order.start_node)
                delivery_time = vehicle.calculate_road_travel_time(order.start_node, order.end_node)
            
            total_completion_time = self.time_step + pickup_time + delivery_time
            ddl_margin = order.deadline - total_completion_time
            
            if ddl_margin >= 0:
                return True
            else:
                return False
                
        except Exception as e:
            return False

    def _ddl_prefilter_by_straight_line(self, vehicle, order):
        try:
            pickup_distance = self._calculate_euclidean_distance(vehicle.current_node, order.start_node)
            delivery_distance = self._calculate_euclidean_distance(order.start_node, order.end_node)
            total_distance = pickup_distance + delivery_distance
            
            if vehicle.vehicle_type == 'uav':
                max_speed = 120.0
            else:
                max_speed = 90.0
            
            ideal_min_time = total_distance / max_speed
            available_time = order.deadline - self.time_step
            
            safety_margin = 1.2
            
            if ideal_min_time * safety_margin <= available_time:
                return True
            else:
                return False
                
        except Exception as e:
            return True
    
    def _is_order_urgent(self, order):
        time_remaining = order.deadline - self.time_step
        return time_remaining <= 10

    def _perform_hierarchical_order_assignment(self, action: tuple) -> int:
        if not self.active_orders:
            return 0
        
        dispatch_decision, uav_weights, carrier_weights = action
        
        available_uavs = [v for v in self.vehicles 
                        if v.vehicle_type == 'uav' and 
                        self.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE]
        available_carriers = [v for v in self.vehicles 
                             if v.vehicle_type == 'carrier' and 
                             self.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE]
        
        active_orders_list = list(self.active_orders.values())[:self.max_concurrent_orders]
        
        if not active_orders_list:
            return 0
        
        sorted_uavs = sorted(available_uavs, key=lambda v: v.vehicle_id)
        sorted_carriers = sorted(available_carriers, key=lambda v: v.vehicle_id)
        sorted_orders = sorted(active_orders_list, key=lambda o: o.order_id)
        
        num_orders = len(sorted_orders)
        num_uavs = len(sorted_uavs)
        num_carriers = len(sorted_carriers)
        
        uav_weight_matrix = uav_weights.reshape(self.num_uavs, 15)[:num_uavs, :num_orders] if num_uavs > 0 else np.zeros((0, num_orders))
        carrier_weight_matrix = carrier_weights.reshape(self.num_ground_vehicles, 15)[:num_carriers, :num_orders] if num_carriers > 0 else np.zeros((0, num_orders))
        
        assignment_count = 0
        used_orders = set()
        
        for o_idx, order in enumerate(sorted_orders):
            if o_idx in used_orders:
                continue
            
            assign_to_uav = dispatch_decision[o_idx] > 0.5 if o_idx < len(dispatch_decision) else False
            
            best_vehicle = None
            best_score = 0.0
            
            if assign_to_uav and num_uavs > 0:
                for v_idx, uav in enumerate(sorted_uavs):
                    if self.vehicle_states.get(uav.vehicle_id) != VehicleState.IDLE:
                        continue
                    score = uav_weight_matrix[v_idx, o_idx] if v_idx < uav_weight_matrix.shape[0] and o_idx < uav_weight_matrix.shape[1] else 0.0
                    # Feasibility is expensive for ground routes. Scores that
                    # cannot beat the incumbent cannot affect the argmax, so
                    # rejecting them first is exactly semantics-preserving.
                    if score <= best_score or score <= 0.3:
                        continue
                    if not self._can_vehicle_handle_order(uav, order):
                        continue
                    if score > best_score:
                        best_score = score
                        best_vehicle = uav
                        
            elif not assign_to_uav and num_carriers > 0:
                for v_idx, carrier in enumerate(sorted_carriers):
                    if self.vehicle_states.get(carrier.vehicle_id) != VehicleState.IDLE:
                        continue
                    score = carrier_weight_matrix[v_idx, o_idx] if v_idx < carrier_weight_matrix.shape[0] and o_idx < carrier_weight_matrix.shape[1] else 0.0
                    if score <= best_score or score <= 0.3:
                        continue
                    if not self._can_vehicle_handle_order(carrier, order):
                        continue
                    if score > best_score:
                        best_score = score
                        best_vehicle = carrier
            
            if best_vehicle is not None:
                if self.assign_order_to_vehicle(best_vehicle.vehicle_id, order):
                    assignment_count += 1
                    used_orders.add(o_idx)
        
        self.total_orders_matched += assignment_count
        
        return assignment_count

    def _can_vehicle_handle_order(self, vehicle, order):
        if not self._can_complete_order_within_ddl(vehicle, order):
            return False
        
        if vehicle.vehicle_type == "uav":
            if hasattr(vehicle, "can_complete_three_phase_mission"):
                can_handle, reason = vehicle.can_complete_three_phase_mission(order)
                return can_handle
            elif hasattr(vehicle, "can_deliver_order_safely"):
                road_order = RoadNetworkOrder(
                    order.order_id, order.start_node, order.end_node,
                    order.start_time, order.deadline, order.weight
                )
                return vehicle.can_deliver_order_safely(road_order)
        else:
            return True
        
        return True

    def _update_vehicle_behaviors(self):
        for vehicle in self.vehicles:
            vehicle_id = vehicle.vehicle_id
            current_state = self.vehicle_states.get(vehicle_id, VehicleState.IDLE)
            
            if current_state == VehicleState.ASSIGNED:
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_pickup_movement(vehicle, assigned_order)
                    
            elif current_state == VehicleState.DELIVERING:
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_delivery_movement(vehicle, assigned_order)

            elif current_state == VehicleState.FLYING_TO_CHARGE:
                self._handle_charging_flight(vehicle)

    def _handle_charging_flight(self, vehicle):
        if not (hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav'):
            return
        
        charge_result = vehicle.return_to_charging_station()
        
        if charge_result == True:
            self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
            self.charging_events += 1
        elif charge_result == "flying_to_charge":
            pass
        else:
            self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE

    def _check_ddl_timeouts(self):
        expired_vehicles = []
        
        for vehicle_id, order in list(self.vehicle_assigned_orders.items()):
            if order.deadline <= self.time_step:
                vehicle_state = self.vehicle_states.get(vehicle_id)

                if vehicle_state == VehicleState.DELIVERING:
                    self.vehicle_complete_delivery(vehicle_id)
                else:
                    self._handle_order_failure(vehicle_id, order)
                    
                expired_vehicles.append(vehicle_id)
        
        return len(expired_vehicles)
    
    def _handle_order_failure(self, vehicle_id, order):
        self.order_statuses[order.order_id] = OrderStatus.EXPIRED
        
        if vehicle_id in self.vehicle_assigned_orders:
            del self.vehicle_assigned_orders[vehicle_id]
        
        if not hasattr(self, "failed_orders"):
            self.failed_orders = []
        self.failed_orders.append(order)

        vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
        if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            need_charge, reason = vehicle.needs_return_to_charge()
            if need_charge:
                charge_result = vehicle.return_to_charging_station()
                
                if charge_result == True:
                    self.vehicle_states[vehicle_id] = VehicleState.IDLE
                    self.charging_events += 1
                elif charge_result == "flying_to_charge":
                    self.vehicle_states[vehicle_id] = VehicleState.FLYING_TO_CHARGE
                else:
                    self.vehicle_states[vehicle_id] = VehicleState.IDLE
            else:
                self.vehicle_states[vehicle_id] = VehicleState.IDLE
        else:
            self.vehicle_states[vehicle_id] = VehicleState.IDLE
    
    def _handle_pickup_movement(self, vehicle, order):
        target_node = order.start_node
        current_node = vehicle.current_node
        
        if self._is_at_pickup_location(vehicle, order):
            if self.vehicle_start_pickup(vehicle.vehicle_id):
                self.vehicle_start_delivery(vehicle.vehicle_id)
            return
        
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            self._handle_uav_direct_flight(vehicle, target_node, "pickup")
        else:
            self._handle_carrier_road_movement(vehicle, target_node, "pickup")
    
    def _handle_delivery_movement(self, vehicle, order):
        target_node = order.end_node
        current_node = vehicle.current_node
        
        if self._is_at_delivery_location(vehicle, order):
            self.vehicle_complete_delivery(vehicle.vehicle_id)
            return
        
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            self._handle_uav_direct_flight(vehicle, target_node, "delivery")
        else:
            self._handle_carrier_road_movement(vehicle, target_node, "delivery")
    
    def _is_at_pickup_location(self, vehicle, order):
        return vehicle.current_node == order.start_node
    
    def _is_at_delivery_location(self, vehicle, order):
        return vehicle.current_node == order.end_node
    
    def get_state(self) -> np.ndarray:
        state = []
        
        MAX_ORDERS = 15
        active_orders_list = list(self.active_orders.values())[:MAX_ORDERS]
        
        for i in range(MAX_ORDERS):
            if i < len(active_orders_list):
                order = active_orders_list[i]
                state.extend([
                    order.start_node / 600000.0,
                    order.end_node / 600000.0,
                    order.start_time / self.max_time_steps,
                    order.deadline / self.max_time_steps,
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
        
        uavs = [v for v in self.vehicles if v.vehicle_type == 'uav']
        for i in range(self.num_uavs):
            if i < len(uavs):
                uav = uavs[i]
                vehicle_state = self.vehicle_states.get(uav.vehicle_id, VehicleState.IDLE)
                is_available = 1.0 if vehicle_state == VehicleState.IDLE else 0.0
                battery_ratio = getattr(uav, 'battery_level', 1) / getattr(uav, 'battery_capacity', 1)
                
                state.extend([
                    uav.current_node / 600000.0,
                    getattr(uav, 'speed', 120) / 120.0,
                    is_available,
                    battery_ratio
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
        
        carriers = [v for v in self.vehicles if v.vehicle_type == 'carrier']
        for i in range(self.num_ground_vehicles):
            if i < len(carriers):
                carrier = carriers[i]
                vehicle_state = self.vehicle_states.get(carrier.vehicle_id, VehicleState.IDLE)
                is_available = 1.0 if vehicle_state == VehicleState.IDLE else 0.0
                
                state.extend([
                    carrier.current_node / 600000.0,
                    getattr(carrier, 'speed', 90) / 120.0,
                    is_available
                ])
            else:
                state.extend([0.0, 0.0, 0.0])
        
        system_info = [
            self.time_step / self.max_time_steps,
            len(self.active_orders) / MAX_ORDERS,
            len(self.completed_orders) / max(self.total_orders_generated, 1),
            sum(1 for v in self.vehicles if v.vehicle_id in self.vehicle_assigned_orders) / len(self.vehicles),
            self.total_orders_matched / max(self.total_orders_generated, 1),
            self.uav_energy_failures / max(self.total_orders_generated, 1),
            self.charging_events / max(self.time_step, 1),
            len([v for v in self.vehicles if v.vehicle_type == 'uav']) / len(self.vehicles)
        ]
        state.extend(system_info)
        
        state_array = np.array(state, dtype=np.float32)
        order_dim = 15 * 4
        uav_dim = self.num_uavs * 4
        carrier_dim = self.num_ground_vehicles * 3
        system_dim = 8
        expected_dim = order_dim + uav_dim + carrier_dim + system_dim
        if len(state_array) != expected_dim:
            if len(state_array) < expected_dim:
                state_array = np.pad(state_array, (0, expected_dim - len(state_array)), 'constant')
            else:
                state_array = state_array[:expected_dim]
        
        return state_array
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        new_orders = self.generate_orders()
        expired_count = self._update_order_status()
        assignment_count = self._perform_hierarchical_order_assignment(action)
        timeout_count = self._check_ddl_timeouts()
        self._update_vehicle_behaviors()
        pickup_count, delivery_count = self._execute_automatic_vehicle_behaviors()
        
        reward = self._calculate_platform_reward(pickup_count, delivery_count, assignment_count)
        
        done = self.time_step >= self.max_time_steps - 1
        
        self.time_step += 1
        
        next_states = self.get_state()
        
        info = {
            'total_orders': self.total_orders_generated,
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'total_generated': self.total_orders_generated,
            'total_completed': self.total_orders_completed,
            'total_matched': self.total_orders_matched,
            'delivery_rate': self.total_orders_completed / max(self.total_orders_generated, 1),
            'match_rate': self.total_orders_matched / max(self.total_orders_generated, 1),
            'new_orders_count': len(new_orders),
            'pickup_count': pickup_count,
            'delivery_count': delivery_count,
            'expired_count': expired_count,
            'timeout_count': timeout_count,
            'path_planning_failures': self.path_planning_failures,
            'uav_energy_failures': self.uav_energy_failures,
            'charging_events': self.charging_events,
            'uav_deliveries': self.episode_uav_deliveries,
            'carrier_deliveries': self.episode_carrier_deliveries,
            'total_uav_deliveries': self.total_uav_deliveries,
            'total_carrier_deliveries': self.total_carrier_deliveries,
        }
        
        return next_states, reward, done, info
    
    def _execute_automatic_vehicle_behaviors(self) -> Tuple[int, int]:
        return self.episode_pickups, self.episode_deliveries
    
    def _calculate_platform_reward(self, pickup_count: int, delivery_count: int, assignment_count: int) -> float:
        reward = 0.0
        
        reward += 1.0
        reward += pickup_count * 3.0
        reward += delivery_count * 10.0
        reward += assignment_count * 1.5
        
        if len(self.active_orders) > 0:
            if self.total_orders_matched > 0:
                order_processing_rate = (pickup_count + delivery_count) / self.total_orders_matched
                reward += order_processing_rate * 10
        
        progress_bonus = (1.0 - self.time_step / self.max_time_steps) * 0.2
        reward += progress_bonus
        
        return reward

    def _update_order_status(self) -> int:
        expired_orders = []
        for order in self.active_orders.values():
            if self.time_step > order.deadline:
                expired_orders.append(order.order_id)
        
        for order_id in expired_orders:
            del self.active_orders[order_id]
        
        return len(expired_orders)
    
    def _handle_uav_direct_flight(self, vehicle, target_node, target_type):
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        current_payload = 0.5 if (hasattr(vehicle, 'vehicle_state') and 
                                 self.vehicle_states.get(vehicle.vehicle_id) == VehicleState.DELIVERING) else 0.0
        
        if hasattr(vehicle, 'calculate_flight_time_to_node') and hasattr(vehicle, 'execute_flight_phase'):
            flight_result, flight_reason = vehicle.execute_flight_phase(target_node, current_payload, "energy_managed_flight")
            
            if flight_result == True:
                pass
            elif flight_result == "flying":
                pass
            else:
                self._force_uav_return_to_charge(vehicle)
        else:
            self._handle_uav_legacy_flight(vehicle, target_node, target_type)
    
    def _handle_uav_legacy_flight(self, vehicle, target_node, target_type):
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        distance = self._calculate_euclidean_distance(current_node, target_node)
        uav_speed = 120.0
        flight_time = distance / uav_speed
        time_steps_needed = max(1, int(flight_time * 6))
        
        if not hasattr(vehicle, 'flight_remaining_steps'):
            vehicle.flight_remaining_steps = 0
            vehicle.flight_target = None
        
        if vehicle.flight_target != target_node:
            vehicle.flight_remaining_steps = time_steps_needed
            vehicle.flight_target = target_node
        
        vehicle.flight_remaining_steps -= 1
        
        if vehicle.flight_remaining_steps <= 0:
            vehicle.current_node = target_node
            vehicle.flight_target = None
        else:
            pass
    
    def _force_uav_return_to_charge(self, vehicle):
        self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
        
        if vehicle.vehicle_id in self.vehicle_assigned_orders:
            failed_order = self.vehicle_assigned_orders[vehicle.vehicle_id]
            del self.vehicle_assigned_orders[vehicle.vehicle_id]
            self.uav_energy_failures += 1
        
        vehicle.current_node = vehicle.charging_station_node
        old_level = vehicle.battery_level
        vehicle.battery_level = vehicle.battery_capacity
    
    def _handle_carrier_road_movement(self, vehicle, target_node, target_type):
        result, message = vehicle.execute_road_movement_phase(target_node, f"to_{target_type}")
        
        if result == "moving":
            return
        elif result == True:
            return
        elif result == False:
            if vehicle.vehicle_id in self.vehicle_assigned_orders:
                failed_order = self.vehicle_assigned_orders[vehicle.vehicle_id]
                self._handle_order_failure(vehicle.vehicle_id, failed_order)
    
    def _calculate_euclidean_distance(self, node1, node2):
        return distance_calculator.get_straight_line_distance(node1, node2)
    
    def print_cache_statistics(self):
        print_global_cache_stats()

    def reset(self, day: Optional[int] = None) -> List[np.ndarray]:
        self.time_step = 0
        self.active_orders = {}
        self.completed_orders = []
        
        target_day = int(day if day is not None else 1)
        if not self.load_road_orders(day=target_day):
            raise RuntimeError(f"Failed to load orders for day {target_day}")
        self.loaded_day = target_day
        
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0
        self.uav_energy_failures = 0
        self.charging_events = 0

        self.episode_pickups = 0
        self.episode_deliveries = 0
        
        self.episode_uav_deliveries = 0
        self.episode_carrier_deliveries = 0

        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        
        if hasattr(self, "mapping_node_ids") and self.mapping_node_ids is not None and self.road_graph is not None:
            valid_mapping_nodes = [node for node in self.mapping_node_ids if node in self.road_graph.nodes()]
            if len(valid_mapping_nodes) >= self.num_vehicles:
                start_nodes = np.random.choice(valid_mapping_nodes, self.num_vehicles, replace=False)
            else:
                start_nodes = np.random.choice(valid_mapping_nodes, self.num_vehicles, replace=True)
        else:
            if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= self.num_vehicles:
                start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
            elif self.largest_component_nodes is not None:
                start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=True)
            else:
                available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
                start_nodes = available_nodes
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle.current_node = start_nodes[i]
            vehicle.current_grid = vehicle.current_node
            vehicle.current_load = 0
            if hasattr(vehicle, "load"):
                vehicle.load = 0
            if hasattr(vehicle, "available_time"):
                vehicle.available_time = 0
            if hasattr(vehicle, "assigned_orders"):
                vehicle.assigned_orders = []
            
            if vehicle.vehicle_type == 'uav':
                vehicle.battery_level = vehicle.battery_capacity
                vehicle.orders = []
                vehicle.target_node = None
                vehicle.path = []
                vehicle.path_index = 0
                vehicle.vehicle_state = VehicleState.IDLE
                vehicle.assigned_order = None
                vehicle.pickup_deadline = None
                vehicle.delivery_deadline = None
                vehicle.in_flight = False
                vehicle.flight_target_node = None
                vehicle.flight_start_node = None
                vehicle.remaining_flight_timesteps = 0
                vehicle.flight_payload = 0.0
                vehicle.flight_total_energy = 0.0
                vehicle.is_flying_to_charge = False
                vehicle.total_energy_consumed = 0.0
                vehicle.total_flight_time = 0.0
                vehicle.completed_orders_count = 0
            else:
                vehicle.worked_hours = 0.0
                vehicle.remaining_road_timesteps = 0
                vehicle.is_moving_on_road = False
                vehicle.road_movement_target = None

            self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
        
        return self.get_state()
