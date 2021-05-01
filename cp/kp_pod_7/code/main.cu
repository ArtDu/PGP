#include <bits/stdc++.h>

using namespace std;

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0)


struct vec3 {
    double x;
    double y;
    double z;

    __device__ __host__
    vec3() {}

    __device__ __host__
    vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct polygon {
    vec3 x;
    vec3 y;
    vec3 z;
    uchar4 color;

    __device__ __host__
    polygon() {}

    __device__ __host__
    polygon(vec3 points[], uchar4 color) {
        x = points[0];
        y = points[1];
        z = points[2];
        this->color = color;
    }

    __device__ __host__
    polygon(vec3 a, vec3 b, vec3 c, uchar4 color) {
        x = a;
        y = b;
        z = c;
        this->color = color;
    }
};

__device__ __host__
double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__
vec3 prod(vec3 a, vec3 b) {
    return vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__device__ __host__
vec3 mult_by_number(vec3 a, double num) {
    return vec3(a.x * num,
                a.y * num,
                a.z * num);
}

__device__ __host__
vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    return vec3(v.x / l,
                v.y / l,
                v.z / l);
}

__device__ __host__
vec3 diff(vec3 a, vec3 b) {
    return vec3(a.x - b.x,
                a.y - b.y,
                a.z - b.z);
}

__device__ __host__
vec3 add(vec3 a, vec3 b) {
    return vec3(a.x + b.x,
                a.y + b.y,
                a.z + b.z);
}

__device__ __host__
vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return vec3(a.x * v.x + b.x * v.y + c.x * v.z,
                a.y * v.x + b.y * v.y + c.y * v.z,
                a.z * v.x + b.z * v.y + c.z * v.z);
}

void scene(polygon polygons[], vec3 points[], vec3 color) {
    uchar4 true_color = make_uchar4( color.x * 255, color.y * 255, color.z * 255, 0);

    polygons[0] = polygon(points[0], points[1], points[2], true_color);
    polygons[1] = polygon(points[0], points[2], points[3], true_color);
}

void hexahedron(polygon polygons[], double radius, vec3 center, vec3 color) {
    uchar4 true_color = make_uchar4( color.x * 255, color.y * 255, color.z * 255, 0);

    double a = 2 * radius / sqrt(3);
    vec3 vertex(center.x - a / 2, center.y - a / 2, center.z - a / 2);
    vec3 vertices[] = {
            vec3( vertex.x,     vertex.y,     vertex.z ),
            vec3( vertex.x,     vertex.y + a, vertex.z ),
            vec3( vertex.x + a, vertex.y + a, vertex.z ),
            vec3( vertex.x + a, vertex.y,     vertex.z ),
            vec3( vertex.x,     vertex.y,     vertex.z + a ),
            vec3( vertex.x,     vertex.y + a, vertex.z + a ),
            vec3( vertex.x + a, vertex.y + a, vertex.z + a ),
            vec3( vertex.x + a, vertex.y,     vertex.z + a )
    };

    polygons[2] = polygon(vertices[0], vertices[1], vertices[2], true_color);
    polygons[3] = polygon(vertices[2], vertices[3], vertices[0], true_color);
    polygons[4] = polygon(vertices[6], vertices[7], vertices[3], true_color);
    polygons[5] = polygon(vertices[3], vertices[2], vertices[6], true_color);
    polygons[6] = polygon(vertices[2], vertices[1], vertices[5], true_color);
    polygons[7] = polygon(vertices[5], vertices[6], vertices[2], true_color);
    polygons[8] = polygon(vertices[4], vertices[5], vertices[1], true_color);
    polygons[9] = polygon(vertices[1], vertices[0], vertices[4], true_color);
    polygons[10] = polygon(vertices[3], vertices[7], vertices[4], true_color);
    polygons[11] = polygon(vertices[4], vertices[0], vertices[3], true_color);
    polygons[12] = polygon(vertices[6], vertices[5], vertices[4], true_color);
    polygons[13] = polygon(vertices[4], vertices[7], vertices[6], true_color);
}

void octahedron(polygon polygons[], double radius, vec3 center, vec3 color) {
    uchar4 true_color = make_uchar4( color.x * 255, color.y * 255, color.z * 255, 0);

    vec3 vertices[] = {
            vec3( center.x + radius, center.y,          center.z            ),
            vec3( center.x - radius, center.y,          center.z            ),
            vec3( center.x,          center.y + radius, center.z            ),
            vec3( center.x,          center.y - radius, center.z            ),
            vec3( center.x,          center.y,          center.z + radius   ),
            vec3( center.x,          center.y,          center.z - radius   )
    };

    polygons[14] = polygon(vertices[5], vertices[2], vertices[0], true_color);
    polygons[15] = polygon(vertices[5], vertices[0], vertices[3], true_color);
    polygons[16] = polygon(vertices[5], vertices[3], vertices[1], true_color);
    polygons[17] = polygon(vertices[5], vertices[1], vertices[2], true_color);
    polygons[18] = polygon(vertices[4], vertices[3], vertices[0], true_color);
    polygons[19] = polygon(vertices[4], vertices[1], vertices[3], true_color);
    polygons[20] = polygon(vertices[4], vertices[2], vertices[1], true_color);
    polygons[21] = polygon(vertices[4], vertices[0], vertices[2], true_color);
}

void dodecahedron(polygon polygons[], double radius, vec3 center, vec3 color) {
    uchar4 true_color = make_uchar4( color.x * 255, color.y * 255, color.z * 255, 0);
    double phi = (1 + sqrt(5)) / 2;

    //sorry for that
    vec3 vertices[] = {
            vec3(center.x + (-1/phi / sqrt(3) * radius),   center.y + ( 0               * radius),      center.z + ( phi     / sqrt(3) * radius)     ),
            vec3(center.x + ( 1/phi / sqrt(3) * radius),   center.y + ( 0               * radius),      center.z + ( phi     / sqrt(3) * radius)     ),
            vec3(center.x + (-1     / sqrt(3) * radius),   center.y + ( 1     / sqrt(3) * radius),      center.z + ( 1       / sqrt(3) * radius)     ),
            vec3(center.x + ( 1     / sqrt(3) * radius),   center.y + ( 1     / sqrt(3) * radius),      center.z + ( 1       / sqrt(3) * radius)     ),
            vec3(center.x + ( 1     / sqrt(3) * radius),   center.y + (-1     / sqrt(3) * radius),      center.z + ( 1       / sqrt(3) * radius)     ),
            vec3(center.x + (-1     / sqrt(3) * radius),   center.y + (-1     / sqrt(3) * radius),      center.z + ( 1       / sqrt(3) * radius)     ),
            vec3(center.x + ( 0               * radius),   center.y + (-phi   / sqrt(3) * radius),      center.z + ( 1/phi   / sqrt(3) * radius)     ),
            vec3(center.x + ( 0               * radius),   center.y + ( phi   / sqrt(3) * radius),      center.z + ( 1/phi   / sqrt(3) * radius)     ),
            vec3(center.x + (-phi   / sqrt(3) * radius),   center.y + (-1/phi / sqrt(3) * radius),      center.z + ( 0                 * radius)     ),
            vec3(center.x + (-phi   / sqrt(3) * radius),   center.y + ( 1/phi / sqrt(3) * radius),      center.z + ( 0                 * radius)     ),
            vec3(center.x + ( phi   / sqrt(3) * radius),   center.y + ( 1/phi / sqrt(3) * radius),      center.z + ( 0                 * radius)     ),
            vec3(center.x + ( phi   / sqrt(3) * radius),   center.y + (-1/phi / sqrt(3) * radius),      center.z + ( 0                 * radius)     ),
            vec3(center.x + ( 0               * radius),   center.y + (-phi   / sqrt(3) * radius),      center.z + (-1/phi   / sqrt(3) * radius)     ),
            vec3(center.x + ( 0               * radius),   center.y + ( phi   / sqrt(3) * radius),      center.z + (-1/phi   / sqrt(3) * radius)     ),
            vec3(center.x + ( 1     / sqrt(3) * radius),   center.y + ( 1     / sqrt(3) * radius),      center.z + (-1       / sqrt(3) * radius)     ),
            vec3(center.x + ( 1     / sqrt(3) * radius),   center.y + (-1     / sqrt(3) * radius),      center.z + (-1       / sqrt(3) * radius)     ),
            vec3(center.x + (-1     / sqrt(3) * radius),   center.y + (-1     / sqrt(3) * radius),      center.z + (-1       / sqrt(3) * radius)     ),
            vec3(center.x + (-1     / sqrt(3) * radius),   center.y + ( 1     / sqrt(3) * radius),      center.z + (-1       / sqrt(3) * radius)     ),
            vec3(center.x + ( 1/phi / sqrt(3) * radius),   center.y + ( 0               * radius),      center.z + (-phi     / sqrt(3) * radius)     ),
            vec3(center.x + (-1/phi / sqrt(3) * radius),   center.y + ( 0               * radius),      center.z + (-phi     / sqrt(3) * radius)     )
    };

    polygons[22] = polygon(vertices[4],  vertices[0],  vertices[6],  true_color);
    polygons[23] = polygon(vertices[0],  vertices[5],  vertices[6],  true_color);
    polygons[24] = polygon(vertices[0],  vertices[4],  vertices[1],  true_color);
    polygons[25] = polygon(vertices[0],  vertices[3],  vertices[7],  true_color);
    polygons[26] = polygon(vertices[2],  vertices[0],  vertices[7],  true_color);
    polygons[27] = polygon(vertices[0],  vertices[1],  vertices[3],  true_color);
    polygons[28] = polygon(vertices[10], vertices[1],  vertices[11],  true_color);
    polygons[29] = polygon(vertices[3],  vertices[1],  vertices[10],  true_color);
    polygons[30] = polygon(vertices[1],  vertices[4],  vertices[11],  true_color);
    polygons[31] = polygon(vertices[5],  vertices[0],  vertices[8],  true_color);
    polygons[32] = polygon(vertices[0],  vertices[2],  vertices[9],  true_color);
    polygons[33] = polygon(vertices[8],  vertices[0],  vertices[9],  true_color);
    polygons[34] = polygon(vertices[5],  vertices[8],  vertices[16], true_color);
    polygons[35] = polygon(vertices[6],  vertices[5],  vertices[12], true_color);
    polygons[36] = polygon(vertices[12], vertices[5],  vertices[16], true_color);
    polygons[37] = polygon(vertices[4],  vertices[12], vertices[15], true_color);
    polygons[38] = polygon(vertices[4],  vertices[6],  vertices[12], true_color);
    polygons[39] = polygon(vertices[11], vertices[4],  vertices[15], true_color);
    polygons[40] = polygon(vertices[2],  vertices[13], vertices[17], true_color);
    polygons[41] = polygon(vertices[2],  vertices[7],  vertices[13], true_color);
    polygons[42] = polygon(vertices[9],  vertices[2],  vertices[17], true_color);
    polygons[43] = polygon(vertices[13], vertices[3],  vertices[14], true_color);
    polygons[44] = polygon(vertices[7],  vertices[3],  vertices[13], true_color);
    polygons[45] = polygon(vertices[3],  vertices[10], vertices[14], true_color);
    polygons[46] = polygon(vertices[8],  vertices[17], vertices[19], true_color);
    polygons[47] = polygon(vertices[16], vertices[8],  vertices[19], true_color);
    polygons[48] = polygon(vertices[8],  vertices[9],  vertices[17], true_color);
    polygons[49] = polygon(vertices[14], vertices[11], vertices[18], true_color);
    polygons[50] = polygon(vertices[11], vertices[15], vertices[18], true_color);
    polygons[51] = polygon(vertices[10], vertices[11], vertices[14], true_color);
    polygons[52] = polygon(vertices[12], vertices[19], vertices[18], true_color);
    polygons[53] = polygon(vertices[15], vertices[12], vertices[18], true_color);
    polygons[54] = polygon(vertices[12], vertices[16], vertices[19], true_color);
    polygons[55] = polygon(vertices[19], vertices[13], vertices[18], true_color);
    polygons[56] = polygon(vertices[17], vertices[13], vertices[19], true_color);
    polygons[57] = polygon(vertices[13], vertices[14], vertices[18], true_color);
}



__device__ __host__
uchar4 ray(vec3 pos, vec3 dir, vec3 light_src, vec3 light_color, polygon polygons[], int len) {
    int min_value = -1;
    double ts_min;
    for (int i = 0; i < len; i++) {
        vec3 e1 = diff(polygons[i].y, polygons[i].x);
        vec3 e2 = diff(polygons[i].z, polygons[i].x);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = diff(pos, polygons[i].x);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div;
        if (ts < 0.0)
            continue;
        if (min_value == -1 || ts < ts_min) {
            min_value = i;
            ts_min = ts;
        }
    }

    if (min_value == -1)
        return make_uchar4(0, 0, 0, 0);


    // To calculate light
    pos = add(mult_by_number(dir, ts_min), pos);
    dir = diff(light_src, pos);
    double length = sqrt(dot(dir, dir));
    dir = norm(dir);

    for (int i = 0; i < len; i++) {
        vec3 e1 = diff(polygons[i].y, polygons[i].x);
        vec3 e2 = diff(polygons[i].z, polygons[i].x);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = diff(pos, polygons[i].x);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div;
        if (ts > 0.0 && ts < length && i != min_value) {
            return make_uchar4(0, 0, 0, 0);
        }
    }

    uchar4 color_min = polygons[min_value].color;
    color_min.x = color_min.x * light_color.x;
    color_min.y = color_min.y * light_color.y;
    color_min.z = color_min.z * light_color.z;
    return color_min;
}

void
render_cpu(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, vec3 light_src, vec3 light_color,
           polygon polygons[],
           int len) {
    int i, j;
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, vec3(0.0, 0.0, 1.0)));
    vec3 by = norm(prod(bx, bz));
    for (j = 0; j < h; j++) {
        for (i = 0; i < w; i++) {
            vec3 v;
            v.x = -1.0 + dw * i;
            v.y = (-1.0 + dh * j) * h / w;
            v.z = z;
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(pc, norm(dir), light_src, light_color, polygons, len);
        }
    }
}

__global__ void
render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, vec3 light_src, vec3 light_color,
       polygon polygons[],
       int len) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int i, j;
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, vec3(0.0, 0.0, 1.0)));
    vec3 by = norm(prod(bx, bz));
    for (j = idy; j < h; j += offsety) {
        for (i = idx; i < w; i += offsetx) {
            vec3 v = vec3(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(pc, norm(dir), light_src, light_color, polygons, len);
        }
    }
}


void ssaa_cpu(uchar4 *src, uchar4 *out, int w, int h, int wScale, int hScale) {
    int n = wScale * hScale;
    int x, y, i, j;
    uchar4 p;
    uint4 s;


    for(y = 0; y < h; y += 1) {
        for(x = 0; x < w; x += 1) {

            s = make_uint4(0,0,0,0);

            for (i = 0; i < wScale; ++i) {
                for (j = 0; j < hScale; ++j){
                    p = src[ w * wScale * (y * hScale + j) + (x * wScale + i) ];
                    s.x += p.x;
                    s.y += p.y;
                    s.z += p.z;
                }
            }
            s.x /= n;
            s.y /= n;
            s.z /= n;

            out[y * w + x] = make_uchar4(s.x, s.y, s.z, s.w);
        }
    }
}

__global__
void ssaa(uchar4 *src, uchar4 *out, int w, int h, int wScale, int hScale) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int n = wScale * hScale;
    int x, y, i, j;
    uchar4 p;
    uint4 s;


    for(y = idy; y < h; y += offsety) {
        for(x = idx; x < w; x += offsetx) {

            s = make_uint4(0,0,0,0);

            for (i = 0; i < wScale; ++i) {
                for (j = 0; j < hScale; ++j){
                    p = src[ w * wScale * (y * hScale + j) + (x * wScale + i) ];
                    s.x += p.x;
                    s.y += p.y;
                    s.z += p.z;
                }
            }
            s.x /= n;
            s.y /= n;
            s.z /= n;

            out[y * w + x] = make_uchar4(s.x, s.y, s.z, s.w);
        }
    }
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    bool gpu = true;

    // default values
    int frames = 30;
    char out[256] = "out/%d.data";
    int width = 640, height = 480;
    double fov = 120;

    double  r_center_0 = 2, z_center_0 = 2, f_center_0 = .5,
            A_center_r = 1, A_center_z = 1.5,
            w_center_r = 1, w_center_z =  .5, w_center_f = 1,
            p_center_r = 0, p_center_z = 1.5,
            r_direction_0 = .5, z_direction_0 = .5, f_direction_0 = .1,
            A_direction_r = 2, A_direction_z = .5,
            w_direction_r = 2, w_direction_z = .5, w_direction_f = 2,
            p_direction_r = 0, p_direction_z = 0;

    vec3 hexahedron_center = vec3(4, 0, 0),
            hexahedron_color = vec3(1, 0, 0);
    double hexahedron_radius = 2;

    vec3 octahedron_center = vec3(0, 3, 0),
            octahedron_color = vec3(0, 1, 0);
    double octahedron_radius = 1;

    vec3 dodecahedron_center = vec3(-2, 1, 1),
            dodecahedron_color = vec3(0, 0, 1);
    double dodecahedron_radius = 1;


    vec3 scene_points[] = {vec3(-10, -10, -1),
                           vec3(-10, 10, -1),
                           vec3(10, 10, -1),
                           vec3(10, -10, -1)};
    vec3 scene_color = vec3(0.952, 0.635, 0.070);


    int light = 1;
    vec3 light_src = vec3(100, 100, 100),
            light_color = vec3(1, 1, 1);

    int multiplier = 1;

    // fillers
    double _;
    string _str;


    // check parameters
    if (argc == 1 || argc == 2) {
        if (argc == 2) {
            // check gpu flag
            if ((string(argv[1]) == "--gpu") || string(argv[1]) == "--default") {
                gpu = true;
            } else if (string(argv[1]) == "--cpu") {
                gpu = false;
            } else {
                cerr << "Invalid command line parameter\n";
                cerr << "Expected one of this:\n"
                        "\t--gpu\n"
                        "\t--default\n"
                        "\t--cpu\n";
                exit(1);
            }

            // check input
            if ((string(argv[1]) == "--gpu") || string(argv[1]) == "--cpu") {
                cin >> frames >> out >> width >> height >> fov;

                cin >> r_center_0 >> z_center_0 >> f_center_0
                    >> A_center_r >> A_center_z
                    >> w_center_r >> w_center_z >> w_center_f
                    >> p_center_r >> p_center_z;

                cin >> r_direction_0 >> z_direction_0 >> f_direction_0
                    >> A_direction_r >> A_direction_z
                    >> w_direction_r >> w_direction_z >> w_direction_f
                    >> p_direction_r >> p_direction_z;

                cin >> hexahedron_center.x >> hexahedron_center.y >> hexahedron_center.z
                    >> hexahedron_color.x >> hexahedron_color.y >> hexahedron_color.z
                    >> hexahedron_radius >> _ >> _ >> _;

                cin >> octahedron_center.x >> octahedron_center.y >> octahedron_center.z
                    >> octahedron_color.x >> octahedron_color.y >> octahedron_color.z
                    >> octahedron_radius >> _ >> _ >> _;

                cin >> dodecahedron_center.x >> dodecahedron_center.y >> dodecahedron_center.z
                    >> dodecahedron_color.x >> dodecahedron_color.y >> dodecahedron_color.z
                    >> dodecahedron_radius >> _ >> _ >> _;

                cin >> scene_points[0].x >> scene_points[0].y >> scene_points[0].z
                    >> scene_points[1].x >> scene_points[1].y >> scene_points[1].z
                    >> scene_points[2].x >> scene_points[2].y >> scene_points[2].z
                    >> scene_points[3].x >> scene_points[3].y >> scene_points[3].z
                    >> _str;
                cin >> scene_color.x >> scene_color.y >> scene_color.z >> _;

                cin >> light;
                assert(light == 1);
                cin >> light_src.x >> light_src.y >> light_src.z
                    >> light_color.x >> light_color.y >> light_color.z
                    >> _ >> multiplier;

            }
        }
    } else {
        cerr << "Wrong number of command line parameters, expected extra one or no one\n";
        exit(1);
    }


    int polygons_sz = 58;
    polygon polygons[polygons_sz],
            *cuda_polygons;

    uchar4  *data = (uchar4 *) malloc(multiplier * multiplier * width * height * sizeof(uchar4)),
            *ssaa_data = (uchar4 *) malloc(width * height * sizeof(uchar4)),
            *cuda_data,
            *ssaa_cuda_data;

    if (gpu) {
        CSC(cudaMalloc((polygon **) (&cuda_polygons), polygons_sz * sizeof(polygon)));
        CSC(cudaMalloc((uchar4 * *)(&cuda_data),      multiplier * multiplier * width * height * sizeof(uchar4)));
        CSC(cudaMalloc((uchar4 * *)(&ssaa_cuda_data), width * height * sizeof(uchar4)));
    }

    // fill polygons
    hexahedron(polygons, hexahedron_radius, hexahedron_center, hexahedron_color);
    octahedron(polygons, octahedron_radius, octahedron_center, octahedron_color);
    dodecahedron(polygons, dodecahedron_radius, dodecahedron_center, dodecahedron_color);
    scene(polygons, scene_points, scene_color);

    if (gpu) {
        CSC(cudaMemcpy(cuda_polygons, polygons, polygons_sz * sizeof(polygon), cudaMemcpyHostToDevice));
    }


    vec3 pc, pv;
    char buff[256];


    for (int iter = 0; iter < frames; ++iter) {
        double step = 2 * M_PI * iter / frames;

        double r_center = A_center_r * sin(w_center_r * step + p_center_r) + r_center_0;
        double z_center = A_center_z * sin(w_center_z * step + p_center_z) + z_center_0;
        double f_center = w_center_f * step + f_center_0;

        double r_direction = A_direction_r * sin(w_direction_r * step + p_direction_r) + r_direction_0;
        double z_direction = A_direction_z * sin(w_direction_z * step + p_direction_z) + z_direction_0;
        double f_direction = w_direction_f * step + f_direction_0;

        pc.x = cos(f_center) * r_center;
        pc.y = sin(f_center) * r_center;
        pc.z = z_center;

        pv.x = cos(f_direction) * r_direction;
        pv.y = sin(f_direction) * r_direction;
        pv.z = z_direction;

        // time to process one frame
        cudaEvent_t start, stop;
        float gpu_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        if (gpu) {
            render<<<dim3(16, 16), dim3(16, 16)>>>
                    (pc, pv, width * multiplier, height * multiplier, fov, cuda_data, light_src, light_color, cuda_polygons, polygons_sz);
            CSC(cudaGetLastError());

            ssaa<<<dim3(16, 16), dim3(16, 16)>>>
                    (cuda_data, ssaa_cuda_data, width, height, multiplier, multiplier);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(data, ssaa_cuda_data, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
        } else {
            render_cpu
                    (pc, pv, width * multiplier, height * multiplier, fov, data, light_src, light_color, polygons, polygons_sz);
            ssaa_cpu
                    (data, ssaa_data, width, height, multiplier, multiplier);
            memcpy(data, ssaa_data, sizeof(uchar4) * width * height);
        }

        // time to process one frame
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);

        sprintf(buff, out, iter);

        cerr << iter << "\t" << gpu_time << "\t"  << width * height * multiplier * multiplier << endl;
        FILE *out = fopen(buff, "w");
        fwrite(&width, sizeof(int), 1, out);
        fwrite(&height, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), width * height, out);
        fclose(out);
    }

    if (gpu) {
        CSC(cudaFree(cuda_data));
        CSC(cudaFree(ssaa_cuda_data));
        CSC(cudaFree(cuda_polygons));
    }
    free(data);
    free(ssaa_data);
    return 0;
}