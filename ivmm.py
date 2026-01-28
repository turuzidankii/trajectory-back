import math
import networkx as nx
from shapely.geometry import Point
from road_network import road_network_service

class SimpleIVMM:
    def __init__(self, search_radius=100, w_dist=0.6, w_heading=0.4):
        """
        简单 IVMM 算法实现
        :param search_radius: 搜路半径 (米)
        :param w_dist: 距离权重 (0.0 - 1.0)
        :param w_heading: 方向权重 (0.0 - 1.0)
        """
        self.search_radius = search_radius
        self.w_dist = w_dist
        self.w_heading = w_heading
        self.graph = road_network_service.graph

    def _get_heading(self, p1, p2):
        """计算两点间的航向角 (0-360度)"""
        dx = p2['lon'] - p1['lon']
        dy = p2['lat'] - p1['lat']
        if dx == 0 and dy == 0: return 0
        angle = math.degrees(math.atan2(dy, dx))
        return angle if angle >= 0 else angle + 360

    def _get_road_heading(self, geom):
        """计算路段几何的总体走向 (简化版)"""
        if not geom or len(geom.coords) < 2:
            return 0
        # 取首尾点计算方向 (也可以取投影点前后的微切线，这里简化处理)
        # shapely coords 是 (lon, lat)
        start_lon, start_lat = geom.coords[0]
        end_lon, end_lat = geom.coords[-1]
        
        dx = end_lon - start_lon
        dy = end_lat - start_lat
        angle = math.degrees(math.atan2(dy, dx))
        return angle if angle >= 0 else angle + 360

    def _heading_similarity(self, gps_heading, road_heading):
        """计算两个角度的相似度 (0.0 - 1.0)"""
        diff = abs(gps_heading - road_heading)
        # 处理 359度和1度的跨越情况
        diff = min(diff, 360 - diff)
        
        # 允许 90 度以内的误差
        if diff > 90:
            return 0.01
        
        # 越接近 0 度，分数越高 (余弦相似度变体)
        return math.cos(math.radians(diff))

    def _gaussian_score(self, dist, sigma=20):
        """距离得分 (高斯分布)"""
        return math.exp(-0.5 * (dist / sigma) ** 2)

    def match(self, trajectory_points):
        """
        执行匹配
        """
        layers = []
        
        # 1. 候选点搜索与打分
        for i, pt in enumerate(trajectory_points):
            cands = road_network_service.get_candidates(pt['lat'], pt['lon'], self.search_radius)
            
            # 计算 GPS 瞬时航向 (利用前后点)
            gps_heading = 0
            if i < len(trajectory_points) - 1:
                next_pt = trajectory_points[i+1]
                gps_heading = self._get_heading(pt, next_pt)
            elif i > 0:
                prev_pt = trajectory_points[i-1]
                gps_heading = self._get_heading(prev_pt, pt)
            
            scored_cands = []
            if cands:
                for cand in cands:
                    # A. 距离得分
                    s_dist = self._gaussian_score(cand['dist_m'])
                    
                    # B. 方向得分
                    # 注意：road_network.py 的 get_candidates 返回了 'geometry'
                    road_heading = self._get_road_heading(cand['geometry'])
                    
                    # 道路是双向的，所以要检查两个方向 (正向和反向)
                    score_forward = self._heading_similarity(gps_heading, road_heading)
                    score_backward = self._heading_similarity(gps_heading, (road_heading + 180) % 360)
                    s_heading = max(score_forward, score_backward)
                    
                    # C. 综合得分
                    total_score = (self.w_dist * s_dist) + (self.w_heading * s_heading)
                    
                    cand['score'] = total_score
                    scored_cands.append(cand)
            
            # 如果没找到候选点，存空列表
            layers.append(scored_cands)

        if not layers: return []

        # 2. 简单的序列连接 (简化版 Viterbi)
        # 这里为了"简单"，我们不计算复杂的拓扑转移概率，而是基于得分为主，连通性为辅
        
        final_path = []
        
        # 初始化：选择第一帧得分最高的点
        if not layers[0]: return []
        current_best = max(layers[0], key=lambda x: x['score'])
        final_path.append(current_best)
        
        # 贪心 + 简单回溯
        for t in range(1, len(layers)):
            curr_cands = layers[t]
            if not curr_cands: continue
            
            prev_cand = final_path[-1]
            best_cand = None
            max_combined_score = -1
            
            for cand in curr_cands:
                # 连通性惩罚：如果跟上一个点不是同一条路，且距离很远，扣分
                connectivity_score = 1.0
                
                # 简单判断：如果 ID 不同，视为切换道路
                if str(cand['edge_id']) != str(prev_cand['edge_id']):
                    connectivity_score = 0.6 # 切换道路有惩罚
                    
                    # 进阶：可以在这里用 networkx 查是否相连，为了速度暂时省略
                
                # 最终得分 = 自身得分 * 连通性系数
                combined = cand['score'] * connectivity_score
                
                if combined > max_combined_score:
                    max_combined_score = combined
                    best_cand = cand
            
            if best_cand:
                final_path.append(best_cand)
            else:
                # 如果所有点都被惩罚没了(极少见)，取自身分最高的
                final_path.append(max(curr_cands, key=lambda x: x['score']))

        # 3. 提取结果路段 ID 用于生成几何
        # 为了适配 algorithms.py 的逻辑，我们需要构造 (u, v) 格式的边列表
        # 但这里我们直接返回候选对象列表，在 algorithms.py 里处理更方便
        return final_path