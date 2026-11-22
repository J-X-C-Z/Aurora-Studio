//! OpenCV 4.5.5 `Subdiv2D` subset used by PixCake Type1016 and Type1012.
//!
//! The PE calls `0x61225C0 -> 0x6123820 xN -> 0x6124010 -> 0x6124030`.
//! Triangle order is the quad-edge scan order, not a sorted or cyclic form.
//! This port follows that state machine and emits the same low-16-bit EBO.

use crate::Refusal;

const PREV_AROUND_ORG: i32 = 0x11;
const PREV_AROUND_DST: i32 = 0x33;
const NEXT_AROUND_LEFT: i32 = 0x13;
const PREV_AROUND_LEFT: i32 = 0x20;

const PTLOC_ERROR: i32 = -2;
const PTLOC_OUTSIDE_RECT: i32 = -1;
const PTLOC_INSIDE: i32 = 0;
const PTLOC_VERTEX: i32 = 1;
const PTLOC_ON_EDGE: i32 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    first_edge: i32,
    pt: Point,
}

impl Vertex {
    fn empty() -> Self {
        Self {
            first_edge: 0,
            pt: Point::new(0.0, 0.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QuadEdge {
    next: [i32; 4],
    pt: [i32; 4],
}

impl QuadEdge {
    fn empty() -> Self {
        Self {
            next: [0; 4],
            pt: [0; 4],
        }
    }

    fn indexed(edge: i32) -> Self {
        Self {
            next: [edge, edge + 3, edge + 2, edge + 1],
            pt: [0; 4],
        }
    }
}

struct Subdiv2D {
    vtx: Vec<Vertex>,
    qedges: Vec<QuadEdge>,
    free_qedge: i32,
    free_point: i32,
    recent_edge: i32,
    top_left: Point,
    bottom_right: Point,
}

impl Subdiv2D {
    fn new(rect: (f32, f32, f32, f32)) -> Self {
        let mut subdivision = Self {
            vtx: Vec::new(),
            qedges: Vec::new(),
            free_qedge: 0,
            free_point: 0,
            recent_edge: 0,
            top_left: Point::new(rect.0, rect.1),
            bottom_right: Point::new(rect.0 + rect.2, rect.1 + rect.3),
        };
        subdivision.init_delaunay(rect);
        subdivision
    }

    fn next_edge_from(edge: i32, edges: &[QuadEdge]) -> i32 {
        edges[(edge >> 2) as usize].next[(edge & 3) as usize]
    }

    fn rotate_edge(edge: i32, rotation: i32) -> i32 {
        (edge & !3) + ((edge + rotation) & 3)
    }

    fn sym_edge(edge: i32) -> i32 {
        edge ^ 2
    }

    fn get_edge(&self, edge: i32, next_edge_type: i32) -> i32 {
        let value = self.qedges[(edge >> 2) as usize].next[((edge + next_edge_type) & 3) as usize];
        (value & !3) + ((value + (next_edge_type >> 4)) & 3)
    }

    fn edge_org(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[(edge & 3) as usize]
    }

    fn edge_dst(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[((edge + 2) & 3) as usize]
    }

    fn new_edge(&mut self) -> i32 {
        if self.free_qedge <= 0 {
            self.qedges.push(QuadEdge::empty());
            self.free_qedge = self.qedges.len() as i32 - 1;
        }
        let edge = self.free_qedge * 4;
        self.free_qedge = self.qedges[(edge >> 2) as usize].next[1];
        self.qedges[(edge >> 2) as usize] = QuadEdge::indexed(edge);
        edge
    }

    fn new_point(&mut self, point: Point, first_edge: i32) -> i32 {
        if self.free_point == 0 {
            self.vtx.push(Vertex::empty());
            self.free_point = self.vtx.len() as i32 - 1;
        }
        let vertex = self.free_point;
        self.free_point = self.vtx[vertex as usize].first_edge;
        self.vtx[vertex as usize] = Vertex {
            first_edge,
            pt: point,
        };
        vertex
    }

    fn delete_edge(&mut self, edge: i32) {
        self.splice(edge, self.get_edge(edge, PREV_AROUND_ORG));
        let symmetric = Self::sym_edge(edge);
        self.splice(symmetric, self.get_edge(symmetric, PREV_AROUND_ORG));
        let index = (edge >> 2) as usize;
        self.qedges[index].next[0] = 0;
        self.qedges[index].next[1] = self.free_qedge;
        self.free_qedge = edge >> 2;
    }

    fn set_edge_points(&mut self, edge: i32, org_pt: i32, dst_pt: i32) {
        self.qedges[(edge >> 2) as usize].pt[(edge & 3) as usize] = org_pt;
        self.qedges[(edge >> 2) as usize].pt[((edge + 2) & 3) as usize] = dst_pt;
        self.vtx[org_pt as usize].first_edge = edge;
        self.vtx[dst_pt as usize].first_edge = edge ^ 2;
    }

    fn splice(&mut self, edge_a: i32, edge_b: i32) {
        let a_next = self.qedges[(edge_a >> 2) as usize].next[(edge_a & 3) as usize];
        let b_next = self.qedges[(edge_b >> 2) as usize].next[(edge_b & 3) as usize];
        let a_rot = Self::rotate_edge(a_next, 1);
        let b_rot = Self::rotate_edge(b_next, 1);
        let a_rot_next = self.qedges[(a_rot >> 2) as usize].next[(a_rot & 3) as usize];
        let b_rot_next = self.qedges[(b_rot >> 2) as usize].next[(b_rot & 3) as usize];
        self.qedges[(edge_a >> 2) as usize].next[(edge_a & 3) as usize] = b_next;
        self.qedges[(edge_b >> 2) as usize].next[(edge_b & 3) as usize] = a_next;
        self.qedges[(a_rot >> 2) as usize].next[(a_rot & 3) as usize] = b_rot_next;
        self.qedges[(b_rot >> 2) as usize].next[(b_rot & 3) as usize] = a_rot_next;
    }

    fn connect_edges(&mut self, edge_a: i32, edge_b: i32) -> i32 {
        let edge = self.new_edge();
        self.splice(edge, self.get_edge(edge_a, NEXT_AROUND_LEFT));
        self.splice(Self::sym_edge(edge), edge_b);
        let org = self.edge_dst(edge_a);
        let dst = self.edge_org(edge_b);
        self.set_edge_points(edge, org, dst);
        edge
    }

    fn swap_edges(&mut self, edge: i32) {
        let symmetric = Self::sym_edge(edge);
        let a = self.get_edge(edge, PREV_AROUND_ORG);
        let b = self.get_edge(symmetric, PREV_AROUND_ORG);
        self.splice(edge, a);
        self.splice(symmetric, b);
        let org = self.edge_dst(a);
        let dst = self.edge_dst(b);
        self.set_edge_points(edge, org, dst);
        self.splice(edge, self.get_edge(a, NEXT_AROUND_LEFT));
        self.splice(symmetric, self.get_edge(b, NEXT_AROUND_LEFT));
    }

    fn init_delaunay(&mut self, rect: (f32, f32, f32, f32)) {
        let big_coord = 3.0 * rect.2.max(rect.3);
        let rx = rect.0;
        let ry = rect.1;
        self.vtx.clear();
        self.qedges.clear();
        self.recent_edge = 0;
        self.top_left = Point::new(rx, ry);
        self.bottom_right = Point::new(rx + rect.2, ry + rect.3);

        let pp_a = Point::new(rx + big_coord, ry);
        let pp_b = Point::new(rx, ry + big_coord);
        let pp_c = Point::new(rx - big_coord, ry - big_coord);

        self.vtx.push(Vertex::empty());
        self.qedges.push(QuadEdge::empty());
        self.free_qedge = 0;
        self.free_point = 0;

        let p_a = self.new_point(pp_a, 0);
        let p_b = self.new_point(pp_b, 0);
        let p_c = self.new_point(pp_c, 0);
        let edge_ab = self.new_edge();
        let edge_bc = self.new_edge();
        let edge_ca = self.new_edge();
        self.set_edge_points(edge_ab, p_a, p_b);
        self.set_edge_points(edge_bc, p_b, p_c);
        self.set_edge_points(edge_ca, p_c, p_a);
        self.splice(edge_ab, Self::sym_edge(edge_ca));
        self.splice(edge_bc, Self::sym_edge(edge_ab));
        self.splice(edge_ca, Self::sym_edge(edge_bc));
        self.recent_edge = edge_ab;
    }

    fn is_right_of(&self, point: Point, edge: i32) -> i32 {
        let org = self.vtx[self.edge_org(edge) as usize].pt;
        let dst = self.vtx[self.edge_dst(edge) as usize].pt;
        let area = triangle_area(point, dst, org);
        if area > 0.0 {
            1
        } else if area < 0.0 {
            -1
        } else {
            0
        }
    }

    fn locate(&mut self, point: Point) -> (i32, i32, i32) {
        let mut vertex = 0;
        let max_edges = self.qedges.len() as i32 * 4;
        if self.qedges.len() < 4 {
            return (PTLOC_ERROR, 0, 0);
        }
        if point.x < self.top_left.x
            || point.y < self.top_left.y
            || point.x >= self.bottom_right.x
            || point.y >= self.bottom_right.y
        {
            return (PTLOC_OUTSIDE_RECT, 0, 0);
        }

        let mut edge = self.recent_edge;
        if edge <= 0 {
            return (PTLOC_ERROR, 0, 0);
        }
        let mut location = PTLOC_ERROR;
        let mut right_of_curr = self.is_right_of(point, edge);
        if right_of_curr > 0 {
            edge = Self::sym_edge(edge);
            right_of_curr = -right_of_curr;
        }

        for _ in 0..max_edges {
            let onext_edge = Self::next_edge_from(edge, &self.qedges);
            let dprev_edge = self.get_edge(edge, PREV_AROUND_DST);
            let right_of_onext = self.is_right_of(point, onext_edge);
            let right_of_dprev = self.is_right_of(point, dprev_edge);

            if right_of_dprev > 0 {
                if right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) {
                    location = PTLOC_INSIDE;
                    break;
                }
                right_of_curr = right_of_onext;
                edge = onext_edge;
            } else if right_of_onext > 0 {
                if right_of_dprev == 0 && right_of_curr == 0 {
                    location = PTLOC_INSIDE;
                    break;
                }
                right_of_curr = right_of_dprev;
                edge = dprev_edge;
            } else if right_of_curr == 0
                && self.is_right_of(self.vtx[self.edge_dst(onext_edge) as usize].pt, edge) >= 0
            {
                edge = Self::sym_edge(edge);
            } else {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }

        self.recent_edge = edge;
        if location == PTLOC_INSIDE {
            let org = self.vtx[self.edge_org(edge) as usize].pt;
            let dst = self.vtx[self.edge_dst(edge) as usize].pt;
            let t1 = (point.x - org.x).abs() + (point.y - org.y).abs();
            let t2 = (point.x - dst.x).abs() + (point.y - dst.y).abs();
            let t3 = (org.x - dst.x).abs() + (org.y - dst.y).abs();
            if t1 < f32::EPSILON {
                location = PTLOC_VERTEX;
                vertex = self.edge_org(edge);
                edge = 0;
            } else if t2 < f32::EPSILON {
                location = PTLOC_VERTEX;
                vertex = self.edge_dst(edge);
                edge = 0;
            } else if (t1 < t3 || t2 < t3)
                && triangle_area(point, org, dst).abs() < f32::EPSILON as f64
            {
                location = PTLOC_ON_EDGE;
                vertex = 0;
            }
        }

        if location == PTLOC_ERROR {
            edge = 0;
            vertex = 0;
        }
        (location, edge, vertex)
    }

    fn insert(&mut self, point: Point) -> Result<(i32, i32), &'static str> {
        let (location, mut curr_edge, mut curr_point) = self.locate(point);
        if location == PTLOC_ERROR || location == PTLOC_OUTSIDE_RECT {
            return Err("subdiv2d locate rejected point");
        }
        if location == PTLOC_VERTEX {
            return Ok((curr_point, location));
        }
        if location == PTLOC_ON_EDGE {
            let deleted_edge = curr_edge;
            curr_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
            self.recent_edge = curr_edge;
            self.delete_edge(deleted_edge);
        } else if location != PTLOC_INSIDE {
            return Err("subdiv2d invalid location");
        }

        if curr_edge == 0 {
            return Err("subdiv2d insert reached edge zero");
        }
        curr_point = self.new_point(point, 0);
        let mut base_edge = self.new_edge();
        let first_point = self.edge_org(curr_edge);
        self.set_edge_points(base_edge, first_point, curr_point);
        self.splice(base_edge, curr_edge);

        let mut guard = 0usize;
        loop {
            base_edge = self.connect_edges(curr_edge, Self::sym_edge(base_edge));
            curr_edge = self.get_edge(base_edge, PREV_AROUND_ORG);
            guard += 1;
            if self.edge_dst(curr_edge) == first_point {
                break;
            }
            if guard > self.qedges.len() * 16 + 16 {
                return Err("subdiv2d connect loop did not terminate");
            }
        }

        curr_edge = self.get_edge(base_edge, PREV_AROUND_ORG);
        let max_edges = self.qedges.len() as i32 * 4;
        for _ in 0..max_edges {
            let temp_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
            let temp_dst = self.edge_dst(temp_edge);
            let curr_org = self.edge_org(curr_edge);
            let curr_dst = self.edge_dst(curr_edge);
            if self.is_right_of(self.vtx[temp_dst as usize].pt, curr_edge) > 0
                && is_pt_in_circle3(
                    self.vtx[curr_org as usize].pt,
                    self.vtx[temp_dst as usize].pt,
                    self.vtx[curr_dst as usize].pt,
                    self.vtx[curr_point as usize].pt,
                ) < 0
            {
                self.swap_edges(curr_edge);
                curr_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
            } else if curr_org == first_point {
                break;
            } else {
                curr_edge = self.get_edge(
                    Self::next_edge_from(curr_edge, &self.qedges),
                    PREV_AROUND_LEFT,
                );
            }
        }
        Ok((curr_point, location))
    }

    fn node_qwords(&self) -> i32 {
        self.qedges.len() as i32 * 4
    }

    fn tcap(&self) -> Result<i32, &'static str> {
        let value = (((self.node_qwords() & !3) - 4) >> 1).max(0);
        Ok(value)
    }

    fn triangle_indices(&self) -> Result<Vec<[i32; 3]>, &'static str> {
        let total_edges = self.qedges.len() as i32 * 4;
        let mut edgemask = vec![false; total_edges as usize];
        let mut out = Vec::new();
        for edge in (4..total_edges).step_by(2) {
            if edgemask[edge as usize] {
                continue;
            }
            let v0 = self.edge_org(edge);
            let edge_b = self.get_edge(edge, NEXT_AROUND_LEFT);
            let v1 = self.edge_org(edge_b);
            let edge_c = self.get_edge(edge_b, NEXT_AROUND_LEFT);
            let v2 = self.edge_org(edge_c);
            edgemask[edge as usize] = true;
            edgemask[edge_b as usize] = true;
            edgemask[edge_c as usize] = true;
            if v0 >= 4 && v1 >= 4 && v2 >= 4 {
                if out.len() >= self.tcap()? as usize {
                    return Err("triangle export exceeded Tcap");
                }
                out.push([v0 - 4, v1 - 4, v2 - 4]);
            }
        }
        Ok(out)
    }
}

fn triangle_area(a: Point, b: Point, c: Point) -> f64 {
    (b.x as f64 - a.x as f64) * (c.y as f64 - a.y as f64)
        - (b.y as f64 - a.y as f64) * (c.x as f64 - a.x as f64)
}

fn is_pt_in_circle3(pt: Point, a: Point, b: Point, c: Point) -> i32 {
    let eps = f32::EPSILON as f64 * 0.125;
    let mut value = (a.x as f64 * a.x as f64 + a.y as f64 * a.y as f64) * triangle_area(b, c, pt);
    value -= (b.x as f64 * b.x as f64 + b.y as f64 * b.y as f64) * triangle_area(a, c, pt);
    value += (c.x as f64 * c.x as f64 + c.y as f64 * c.y as f64) * triangle_area(a, b, pt);
    value -= (pt.x as f64 * pt.x as f64 + pt.y as f64 * pt.y as f64) * triangle_area(a, b, c);
    if value > eps {
        1
    } else if value < -eps {
        -1
    } else {
        0
    }
}

/// Triangulates an A-point list in caller order and returns the PixCake EBO.
pub fn triangulate(points: &[[f32; 2]]) -> Result<Vec<u16>, Refusal> {
    if points.len() < 3 {
        return Err(Refusal::MissingInput("subdiv2d points"));
    }
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("subdiv2d points"));
    }
    let mut subdivision = Subdiv2D::new((-100000.0, -100000.0, 200000.0, 200000.0));
    for point in points {
        subdivision
            .insert(Point::new(point[0], point[1]))
            .map_err(Refusal::ValueOutOfRange)?;
    }
    let triangles = subdivision
        .triangle_indices()
        .map_err(Refusal::ShapeMismatch)?;
    let mut ebo = Vec::with_capacity(triangles.len() * 3);
    for triangle in triangles {
        for index in triangle {
            if index < 0 || index >= points.len() as i32 {
                return Err(Refusal::ValueOutOfRange("subdiv2d index"));
            }
            ebo.push(index as u16);
        }
    }
    Ok(ebo)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A27: [[f32; 2]; 27] = [
        [120.0, 240.0],
        [180.0, 260.0],
        [240.0, 240.0],
        [117.142_85, 95.102_04],
        [134.285_7, 91.836_73],
        [151.428_57, 90.204_086],
        [168.571_43, 90.204_086],
        [185.714_28, 91.836_73],
        [230.974_29, 100.651_44],
        [265.740_57, 100.210_304],
        [300.026_3, 95.924_59],
        [333.831_42, 87.794_296],
        [367.155_98, 75.819_44],
        [400.0, 60.0],
        [434.285_7, 65.714_29],
        [468.571_44, 71.428_57],
        [502.857_15, 77.142_85],
        [537.142_9, 82.857_15],
        [571.428_6, 88.571_43],
        [605.714_3, 94.285_71],
        [636.686_77, 98.808_93],
        [653.829_6, 105.951_78],
        [670.096_1, 115.197_914],
        [685.486_2, 126.547_32],
        [760.0, 120.0],
        [820.0, 100.0],
        [880.0, 80.0],
    ];

    const EBO: [u16; 132] = [
        0, 3, 4, 18, 23, 2, 23, 18, 19, 1, 0, 7, 2, 15, 16, 15, 2, 12, 1, 2, 24, 2, 1, 7, 0, 4, 5,
        0, 5, 6, 0, 6, 7, 4, 13, 5, 7, 13, 12, 13, 7, 6, 6, 5, 13, 11, 7, 12, 7, 11, 10, 2, 7, 8,
        7, 10, 9, 2, 8, 9, 8, 7, 9, 2, 9, 10, 2, 10, 11, 2, 11, 12, 12, 14, 15, 14, 12, 13, 13, 26,
        14, 26, 15, 14, 15, 26, 16, 2, 16, 17, 16, 26, 17, 2, 17, 18, 17, 26, 18, 18, 26, 19, 23,
        19, 22, 19, 26, 20, 19, 21, 22, 21, 19, 20, 20, 26, 25, 2, 23, 24, 21, 20, 25, 21, 25, 24,
        22, 21, 24, 23, 22, 24,
    ];

    #[test]
    fn fixture_ebo_matches_pinned_quadedge_order() {
        let actual = triangulate(&A27).unwrap();
        assert_eq!(actual, EBO);
        assert_eq!(actual.len(), 132);
    }

    #[test]
    fn duplicate_points_reuse_one_vertex() {
        let points = [[100.0, 100.0]; 3];
        assert!(triangulate(&points).unwrap().is_empty());
    }
}
