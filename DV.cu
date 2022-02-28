
// includes, openGL
#include <GL/glew.h>
#include <GL/freeglut.h>   // freeglut.h might be a better alternative, if available.

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvfunctional>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <string>

#include "src/algorithms/GPU_PROCLUS.cuh"
#include "src/utils/mem_util.h"


#define REFRESH_DELAY     8 //ms
#define MOUSE_UP     0
#define MOUSE_DOWN     1
#define MOUSE_MOVE     3

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class Plot;

class Table;

void run_proclus();


float sim_dist(float *D, int d, int i, int j) {
    float distance = 0.;
    for (int l = 0; l < d; l++) {
        float def = D[d * i + l] - D[d * j + l];
        distance += def * def;
    }
    return sqrt(distance);
}

//float SilhouetteCoefficient(float *D, int *C, int d, int n, int k) {
//
//    int sizes[k];
//    float avg_d[k];
//    float SC = 0.;
//
//
//    for (int c = 0; c < k; c++) {
//        sizes[c] = 0.;
//    }
//    for (int i = 0; i < n; i++) {
//
//        int c_i = C[i];
//        if (c_i >= 0) {
//            sizes[c_i]++;
//        }
//    }
//
//    for (int i = 0; i < n; i++) {
//
//        for (int c = 0; c < k; c++) {
//            avg_d[c] = 0.;
//        }
//
//        int c_i = C[i];
//        if (c_i < 0) {
//            SC += 1.;
//            continue;
//        }
//
//        for (int j = 0; j < n; j++) {
//            if (i != j) {
//                int c_j = C[j];
//                avg_d[c_j] += sim_dist(D, d, i, j);
//            }
//        }
//        for (int c_j = 0; c_j < k; c_j++) {
//            avg_d[c_j] /= (sizes[c_j] - 1);
//        }
//
//        float a = avg_d[c_i];
//        float b = 0.;
//        bool first = true;
//        for (int c_j = 0; c_j < k; c_j++) {
//            if (c_i != c_j) {
//                if (first) {
//                    first = false;
//                    b = avg_d[c_j];
//                } else if (b > avg_d[c_j]) {
//                    b = avg_d[c_j];
//                }
//            }
//        }
//        float s = (b - a) / max(b, a);
//        SC += s;
//    }
//
//    return SC / n;
//}

float tr(float *A, int d) {
    float sum = 0.;
    for (int i = 0; i < d; i++) {
        sum += A[i * d + i];
    }
    return sum;
}

float CalinskiHarabaszScore(float *D, int *C, int d, int n, int k) {

    // cluster means
    float **c_means = zeros_2d<float>(k, d);
    float *c_sizes = zeros_1d<float>(k);
    for (int p = 0; p < n; p++) {
        int c_p = C[p];
        if (c_p < 0) {
            continue;
        }
        for (int i = 0; i < d; i++) {
            c_means[c_p][i] += D[p * d + i];
        }
        c_sizes[c_p]++;
    }
    for (int c = 0; c < k; c++) {
        for (int i = 0; i < d; i++) {
            c_means[c][i] /= c_sizes[c];
        }
    }

    //data mean
    float *D_mean = zeros_1d<float>(d);
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < d; i++) {
            D_mean[i] += D[p * d + i];
        }
    }
    for (int i = 0; i < d; i++) {
        D_mean[i] /= n;
    }

    //trW
    float trW = 0.;
    for (int p = 0; p < n; p++) {
        int c_p = C[p];
        if (c_p < 0) {
            continue;
        }

        float *x = &D[p * d];
        for (int i = 0; i < d; i++) {
            float diff = x[i] - c_means[c_p][i];
            trW += diff * diff;
        }
    }

    //trB
    float trB = 0.;
    for (int c = 0; c < k; c++) {
        for (int i = 0; i < d; i++) {
            float diff = c_means[c][i] - D_mean[i];
            trB += c_sizes[c] * diff * diff;
        }
    }

    float s = trB / trW * (n - k) / (k - 1);

    //todo delete
    delete c_means;
    delete c_sizes;
    delete D_mean;

    return s;
}

class ThisContext {
public:
    bool use_GPU = true;
    int spf = 0;

    float button_r = 0. / 256.;
    float button_g = 61. / 256.;
    float button_b = 115. / 256.;

    //glColor4f(64. / 256., 128. / 256., 231. / 256., 0.2);
    float select_r = 55. / 256.;
    float select_g = 160. / 256.;
    float select_b = 203. / 256.;


    int window_width = 1700;
    int window_height = 850;
    int n = 0;
    int d = 0;
    int a = 40;
    int b = 10;
    float min_deviation = 0.7;
    int termination_rounds = 5;
    int k = 6;
    int k_center = 6;
    int k_max = 100;
    int l = 7;
    int l_center = 7;
    float *min_values;
    float *max_values;
    float t = 0.;
    int selected_i = 0;
    int selected_j = 1;
    int selected_c = -2;
    Result result;
    std::map <std::pair<int, int>, Result> results;
    GPU_FAST_PROCLUS_C *model = nullptr;

    bool range_selection_started = false;
    int range_x_start = 0;
    int range_x_end = 0;
    int range_y_start = 0;
    int range_y_end = 0;
    float range_x_min = 0;
    float range_x_max = 0;
    float range_y_min = 0;
    float range_y_max = 0;
    int range_x = selected_i;
    int range_y = selected_j;

//    //measures
//    float SC = 0.;
//    float CH = 0.;

    Table *body;

    std::function<void()> tooltip = nullptr;

    // mouse controls
    int mouse_old_x, mouse_old_y;
    int mouse_x, mouse_y;
    int mouse_buttons = 0;

    //points
    GLuint glBuffer;
    struct cudaGraphicsResource *cuBuffer;
    //float4* cuVertex;
    float *d_points_in;
    float *h_points_in;

    std::vector <std::function<void()>> change_listeners;

//    int colors[13 * 3] = {
//            62, 62, 62,
//            31, 120, 180,
//            51, 160, 44,
//            227, 26, 28,
//            255, 127, 0,
//            106, 61, 154,
//            177, 89, 40,
//            166, 206, 227,
//            178, 223, 138,
//            251, 154, 153,
//            253, 191, 111,
//            202, 178, 214,
//            255, 255, 153
//    };
    int colors[11 * 3] = {
            62, 62, 62,
            20, 75, 112,
            31, 96, 27,
            160, 19, 21,
            188, 91, 0,
            68, 39, 99,
//            177, 89, 40,
            166, 206, 227,
            178, 223, 138,
            251, 154, 153,
            253, 191, 111,
            202, 178, 214,
//            255, 255, 153
    };

    ThisContext() {}

    Result get_result(int k_, int l_) {
        if (model == nullptr) {
            model = new GPU_FAST_PROCLUS_C(d_points_in, n, d, k_max, a, b, min_deviation, termination_rounds);
        }

        return model->get_result(k_, l_);
        /*
		std::pair<int, int> key(k_, l_);
		if (results.find(key) == results.end()) {

			int a = 40;
			int b = 10;
			float min_deviation = 0.5;
			int termination_rounds = 20;

			results[key] = GPU_PROCLUS_SAVE(d_points_in, n, d, k_, l_, a, b, min_deviation, termination_rounds, false);
		}

		return results[key];
		*/
    }

    void add_change_listener(std::function<void()> l) {
        change_listeners.push_back(l);
    }

    void changed() {
        for (auto f: (change_listeners)) {
            f();
        }
    }
};

ThisContext context;

class Element {
protected:
    int x_offset, y_offset, width, height;
    std::vector <std::function<void()>> up_listeners;
    std::vector <std::function<void()>> down_listeners;
    std::vector <std::function<void()>> move_listeners;
public:
    int get_x_offset() {
        return x_offset;
    }

    int get_y_offset() {
        return y_offset;
    }

    int get_width() {
        return width;
    }

    int get_height() {
        return height;
    }

    bool intersect(int x, int y) {
        return x_offset <= x && x <= x_offset + width
               && y_offset <= y && y <= y_offset + height;
    }

    void mouse_changed(int x, int y, int state) {
        if (intersect(x, y)) {
            if (state == MOUSE_UP) {
                for (std::function<void()> l: up_listeners) {
                    l();
                }
            } else if (state == MOUSE_DOWN) {
                for (std::function<void()> l: down_listeners) {
                    l();
                }
            } else if (state == MOUSE_MOVE) {
                for (std::function<void()> l: move_listeners) {
                    l();
                }
            }
            on_mouse_changed(x, y, state);
        }
    }

    void add_click_listener(std::function<void()> l) {
        up_listeners.push_back(l);
    }

    void add_down_listener(std::function<void()> l) {
        down_listeners.push_back(l);
    }

    void add_move_listener(std::function<void()> l) {
        move_listeners.push_back(l);
    }

    virtual void build(int x_offset, int y_offset) = 0;

    virtual void paint() = 0;

    virtual void on_mouse_changed(int x, int y, int state) = 0;
};

class Line : Element {
private:
    int x1, y1, x2, y2;
    float r = 0, g = 0, b = 0;
public:
    Line(int x1 = 0, int y1 = 0, int x2 = 0, int y2 = 0) : x1(x1), y1(y1), x2(x2), y2(y2) {
    }

    void setColor(GLfloat new_r, GLfloat new_g, GLfloat new_b) {
        r = new_r;
        g = new_g;
        b = new_b;
    }

    void set_start(int x, int y) {
        x1 = x;
        y1 = y;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x2 - x1;
        height = y2 - y1;
    }

    void paint() {
        glBegin(GL_LINES);
        glLineWidth(1);
        glColor3f(r, g, b);
        glVertex2i(x_offset + x1, y_offset + y1);
        glVertex2i(x_offset + x2, y_offset + y2);
        glEnd();
    }

    void on_mouse_changed(int x, int y, int s) {
    }
};

class Square : Element {

};

class Text : Element {
private:
    std::function<const char *()> get_text = [&]() { return ""; };
    int h_alignment = 0;//left
    int v_alignment = 2;//top
    void *font = GLUT_BITMAP_HELVETICA_10;

    int extra_x_offset = 0;
    int extra_y_offset = 0;

    std::function<void(Text *)> get_color = [&](Text *t) { return; };
public:
    float r = 0.;
    float g = 0.;
    float b = 0.;

    Text(int new_width, int new_height) {
        width = new_width;
        height = new_height;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset + extra_x_offset;
        y_offset = new_y_offset + extra_y_offset;
    }

    void set_extra_x_offset(int x) {
        extra_x_offset = x;
    }

    void set_extra_y_offset(int y) {
        extra_y_offset = y;
    }

    void set_text(std::function<const char *()> f_text) {
        get_text = f_text;
    }

    void set_font(void *new_font) {
        font = new_font;
    }

    void set_color(std::function<void(Text *)> f_color) {
        get_color = f_color;
    }

    void auto_size() {
        const char *text = get_text();
        width = glutBitmapLength(font, (const unsigned char *) text);
        height = glutBitmapHeight(font) * 0.8;
    }

    int get_text_with() {
        const char *text = get_text();
        return glutBitmapLength(font, (const unsigned char *) text);
    }

    int get_text_height() {
        const char *text = get_text();
        return glutBitmapHeight(font) * 0.8;
    }

    void paint() {
        const char *text = get_text();

        int text_width = glutBitmapLength(font, (const unsigned char *) text);
        int text_height = glutBitmapHeight(font) * 0.8;

        int x_extra = 0;
        if (h_alignment == 1) {
            x_extra = width - text_width;
            x_extra /= 2;
        } else if (h_alignment == 2) {
            x_extra = width - text_width;
        }

        int y_extra = 0;
        if (v_alignment == 1) {
            y_extra = height - text_height;
            y_extra /= 2;
        } else if (v_alignment == 2) {
            y_extra = height - text_height;
        }

        get_color(this);
        glColor3f(r, g, b);

        glRasterPos2f(x_offset + x_extra, y_offset + y_extra);

        glutBitmapString(font, (const unsigned char *) text);
        delete text;

        //RenderString(x_offset, y_offset + height, GLUT_BITMAP_HELVETICA_10, text);
    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void v_align_bottom() {
        v_alignment = 0;
    }

    void v_align_center() {
        v_alignment = 1;
    }

    void v_align_top() {
        v_alignment = 2;
    }

    void h_align_left() {
        h_alignment = 0;
    }

    void h_align_center() {
        h_alignment = 1;
    }

    void h_align_right() {
        h_alignment = 2;
    }
};

class Group : Element {
private:
    std::vector<Element *> elements;
public:
    ~Group() {
        /*for (Element* element : elements) {
			delete element;
		}*/
    }

    void on_mouse_changed(int x, int y, int state) {
        for (Element *element: elements) {
            element->mouse_changed(x, y, state);
        }
    }

    void add(Element *element) {
        elements.push_back(element);
    }

    void build(int new_x_offset, int new_y_offset) {

        x_offset = new_x_offset;
        y_offset = new_y_offset;

        int x_min = context.window_width;
        int y_min = context.window_height;
        int x_max = 0;
        int y_max = 0;
        for (Element *element: elements) {
            element->build(x_offset, y_offset);
            /*if (element->get_x_offset() < x_min) {
				x_min = element->get_x_offset();
			}
			if (element->get_y_offset() < y_min) {
				y_min = element->get_y_offset();
			}*/
            if (element->get_x_offset() + element->get_width() > x_max) {
                x_max = element->get_x_offset() + element->get_width();
            }
            if (element->get_y_offset() + element->get_height() > y_max) {
                y_max = element->get_y_offset() + element->get_height();
            }
        }

        width = x_max - x_offset;
        height = y_max - y_offset;
    }

    void paint() {
        for (Element *element: elements) {
            element->paint();
        }
    }
};

class Scale {
private:
    float domain_min, domain_max, range_min, range_max;
public:
    Scale(float domain_min, float domain_max, float range_min, float range_max) : domain_min(domain_min),
                                                                                  domain_max(domain_max),
                                                                                  range_min(range_min),
                                                                                  range_max(range_max) {}

    float get_domain_min() { return domain_min; }

    float get_domain_max() { return domain_max; }

    float get_range_min() { return range_min; }

    float get_range_max() { return range_max; }

    void domain(float min, float max) {
        domain_min = min;
        domain_max = max;
    }

    float convert_to_domain(float x) {
        return domain_min + (domain_max - domain_min) * (x - range_min) / (range_max - range_min);
    }

    float convert_to_range(float x) {
        return range_min + (range_max - range_min) * (x - domain_min) / (domain_max - domain_min);
    }
};

template<typename T>
std::string to_string(const T a_value) {
    std::ostringstream out;
    out << a_value;
    return out.str();
}

const int AXIS_X = 0;
const int AXIS_Y = 1;

class Axis : Element {
private:
    Group *axis_group = new Group();
    Scale *scale;
    int orientation = AXIS_X;
    int tick_count;
    int tick_height = 0;
    int tick_spacing = 50;
    int tick_spacing_min = 50;
    float domain_spacing = 0;
    bool tick_label = false;

    float possible_intervals[3] = {1., 2., 5.};

public:
    Axis(Scale *scale) : scale(scale) {
        tick_count = (scale->get_range_max() - scale->get_range_min()) / tick_spacing + 1;
    }

    ~Axis() {
        //delete axis_group;
    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void enable_tick_label() {
        tick_label = true;
    }

    void left() {
        orientation = AXIS_Y;
    }

    void bottom() {
        orientation = AXIS_X;
    }

    void set_tick_height(int s) {
        tick_height = s;
    }

    float compute_interval() {
        float s = 1.;


        while (s >= scale->get_domain_max() && s >= -scale->get_domain_min()) {
            s /= 10.;
        }

        while (s * 100. < scale->get_domain_max() && s * 100. < -scale->get_domain_min()) {
            s *= 10.;
        }


        float domain_spacing_min = scale->convert_to_domain(tick_spacing_min) - scale->convert_to_domain(0);

        for (int i = 2; i >= 0; i--) {
            if (s * possible_intervals[i] > domain_spacing_min) {
                domain_spacing = s * possible_intervals[i];
            }
        }

        return domain_spacing;

    }

    void build(GLint new_x_offset, GLint new_y_offset) {

        float interval = compute_interval();

        tick_spacing = scale->convert_to_range(interval) - scale->convert_to_range(0);

        tick_count = (scale->get_range_max() - scale->get_range_min()) / tick_spacing + 1;


        x_offset = new_x_offset;
        y_offset = new_y_offset;
        if (orientation == AXIS_X) {
            width = scale->get_range_max();
            height = tick_height;
        } else {
            width = tick_height;
            height = scale->get_range_max();
        }

        Line *axis_line = (orientation == AXIS_X) ?
                          new Line(0, tick_height, scale->get_range_max(), tick_height) :
                          new Line(tick_height, 0, tick_height, scale->get_range_max());
        axis_group->add((Element *) axis_line);

        for (int i = 0; i < tick_count; i++) {
            Line *axis_tick = (orientation == AXIS_X) ?
                              new Line(i * tick_spacing, 0, i * tick_spacing, tick_height) :
                              new Line(0, i * tick_spacing, tick_height, i * tick_spacing);
            axis_group->add((Element *) axis_tick);

            if (tick_label) {
                if (orientation == AXIS_X) {
                    Text *txt = new Text(tick_spacing, tick_height);
                    char str[16];
                    sprintf(str, "%.3f", i * interval);
                    txt->set_text([str, i, interval]() {
                        std::string str = to_string(i * interval);
                        char *cstr = new char[str.length() + 1];
                        strcpy(cstr, str.c_str());
                        return cstr;
                    });
                    txt->auto_size();
                    txt->set_extra_x_offset(i * tick_spacing - ((Element *) txt)->get_width() / 2.);
                    axis_tick->set_start(i * tick_spacing, ((Element *) txt)->get_height());

                    axis_group->add((Element *) txt);
                } else {

                    Text *txt = new Text(tick_spacing, tick_height);
                    char str[16];
                    sprintf(str, "%.3f", i * interval);
                    txt->set_text([str, i, interval]() {
                        std::string str = to_string(i * interval);
                        char *cstr = new char[str.length() + 1];
                        strcpy(cstr, str.c_str());
                        return cstr;
                    });
                    txt->auto_size();
                    txt->set_extra_y_offset(i * tick_spacing - ((Element *) txt)->get_height() / 2.);
                    axis_tick->set_start(((Element *) txt)->get_width(), i * tick_spacing);

                    axis_group->add((Element *) txt);
                }
            }
        }

        /*
		Line* axis_tick = (orientation == AXIS_X) ?
			new Line(scale->get_range_max(), 0, scale->get_range_max(), tick_height) :
			new Line(0, scale->get_range_max(), tick_height, scale->get_range_max());
		axis_group->add((Element*)axis_tick);
		*/

        axis_group->build(new_x_offset, new_y_offset);
    }

    void paint() {
        axis_group->paint();
    }
};


class Cell {
private:
    Element *content = nullptr;
    int width, height;
    int h_alignment = 0;//left
    int v_alignment = 0;//bottom
public:

    Cell(int width, int height) : width(width), height(height) {}

    Cell(int width, int height, Element *content) : width(width), height(height), content(content) {}

    Element *get_content() {
        return content;
    }

    void set_content(Element *e) {
        content = e;
    }

    int get_height() {
        return height;
    }

    int get_width() {
        return width;
    }

    void v_align_bottom() {
        v_alignment = 0;
    }

    void v_align_center() {
        v_alignment = 1;
    }

    void v_align_top() {
        v_alignment = 2;
    }

    void h_align_left() {
        h_alignment = 0;
    }

    void h_align_center() {
        h_alignment = 1;
    }

    void h_align_right() {
        h_alignment = 2;
    }

    void build(int x_offset, int y_offset) {
        if (content != nullptr) {
            int x_extra = 0;

            if (h_alignment == 1) {
                x_extra = width - content->get_width();
                x_extra /= 2;
            } else if (h_alignment == 2) {
                x_extra = width - content->get_width();
            }

            int y_extra = 0;
            if (v_alignment == 1) {
                y_extra = height - content->get_height();
                y_extra /= 2;
            } else if (v_alignment == 2) {
                y_extra = height - content->get_height();
            }

            content->build(x_offset + x_extra, y_offset + y_extra);
        }
    }
};

class Row {
private:
    std::vector<Cell *> cells;
public:
    Row() {}

    Row(Cell *cell) {
        cells.push_back(cell);
    }

    void add(Cell *cell) {
        cells.push_back(cell);
    }

    std::vector<Cell *> get_cells() { return cells; }
};

class Table : Element {
private:
    std::vector<Row *> rows;
    int spacing = 0;
public:
    void on_mouse_changed(int x, int y, int state) {
        for (Row *row: rows) {
            for (Cell *cell: row->get_cells()) {
                Element *content = cell->get_content();
                if (content != nullptr) {
                    content->mouse_changed(x, y, state);
                }
            }
        }
    }

    void add(Row *row) {
        rows.push_back(row);
    }

    void set_spacing(int s) {
        spacing = s;
    }

    void build(GLint new_x_offset, GLint new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;

        std::vector<int> row_heights(rows.size(), 0);
        std::vector<int> col_widths;

        int i = 0;
        for (Row *row: rows) {
            int j = 0;
            for (Cell *cell: row->get_cells()) {

                if (cell->get_height() > row_heights[i]) {
                    row_heights[i] = cell->get_height();
                }

                if (j <= col_widths.size()) {
                    col_widths.push_back(0);
                }

                if (cell->get_width() > col_widths[j]) {
                    col_widths[j] = cell->get_width();
                }

                j++;
            }

            i++;
        }

        width = 0;
        for (int w: col_widths) {
            width += w;
        }
        width += spacing * (col_widths.size() - 1);

        height = 0;
        for (int h: row_heights) {
            height += h;
        }
        height += spacing * (row_heights.size() - 1);

        int y_offset_extra = 0;
        for (int i = rows.size() - 1; i >= 0; i--) {
            Row *row = rows[i];
            std::vector < Cell * > cells = row->get_cells();

            int x_offset_extra = 0;
            for (int j = 0; j < cells.size(); j++) {
                Cell *cell = cells[j];

                cell->build(x_offset + x_offset_extra, y_offset + y_offset_extra);
                /*
				Element* content = cell->get_content();
				if (content != nullptr) {
					content->build(x_offset + x_offset_extra, y_offset + y_offset_extra);
				}
				*/
                x_offset_extra += col_widths[j] + spacing;
            }

            y_offset_extra += row_heights[i] + spacing;
        }
    }

    void paint() {
        for (Row *row: rows) {
            for (Cell *cell: row->get_cells()) {
                Element *content = cell->get_content();
                if (content != nullptr) {
                    content->paint();
                }
            }
        }
    }
};

__global__
void kernel_scale(float *d_point_out, float *d_point_in, int samples, int d, int x_dim, int y_dim, int sample_interval,
                  int entries_per_thread,
                  GLint x_offset, GLint y_offset,
                  float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                  float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {

    for (int j_ = (threadIdx.x + blockDim.x * blockIdx.x) * entries_per_thread;
         j_ < samples; j_ += blockDim.x * gridDim.x * entries_per_thread) {
        for (int j = j_; j < min(samples, j_ + entries_per_thread); j++) {
            int p = j * sample_interval;
            d_point_out[j * 2 + 0] = x_offset + x_range_min +
                                     (x_range_max - x_range_min) * (d_point_in[p * d + x_dim] - x_domain_min) /
                                     (x_domain_max - x_domain_min);
            d_point_out[j * 2 + 1] = y_offset + y_range_min +
                                     (y_range_max - y_range_min) * (d_point_in[p * d + y_dim] - y_domain_min) /
                                     (y_domain_max - y_domain_min);
        }
    }
}

__global__
void kernel_color(float *__restrict__ d_colors, const float *__restrict__ d_points, const int *__restrict__ d_C,
                  const int *__restrict__ d_C_sel, const bool *__restrict__ d_D, const int samples, const int d,
                  const int k, const int x_dim, const int y_dim, const int sample_interval,
                  const int entries_per_thread,
                  const int selected_c, const float range_x_min, const float range_x_max, const float range_y_min,
                  const float range_y_max, const int range_x, const int range_y) {

    int color_components = 4;

//    const float colors[13 * 3] = {
//            62. / 256., 62. / 256., 62. / 256.,
//            31. / 256., 120. / 256., 180. / 256.,
//            51. / 256., 160. / 256., 44. / 256.,
//            227. / 256., 26. / 256., 28. / 256.,
//            255. / 256., 127. / 256., 0. / 256.,
//            106. / 256., 61. / 256., 154. / 256.,
//            177. / 256., 89. / 256., 40. / 256.,
//            166. / 256., 206. / 256., 227. / 256.,
//            178. / 256., 223. / 256., 138. / 256.,
//            251. / 256., 154. / 256., 153. / 256.,
//            253. / 256., 191. / 256., 111. / 256.,
//            202. / 256., 178. / 256., 214. / 256.,
//            255. / 256., 255. / 256., 153. / 256.
//    };


    const float colors[11 * 3] = {
            62, 62, 62,
            20, 75, 112,
            31, 96, 27,
            160, 19, 21,
            188, 91, 0,
            68, 39, 99,
//            177, 89, 40,
            166, 206, 227,
            178, 223, 138,
            251, 154, 153,
            253, 191, 111,
            202, 178, 214,
//            255, 255, 153
    };


    for (int j_ = (threadIdx.x + blockDim.x * blockIdx.x) * entries_per_thread;
         j_ < samples; j_ += blockDim.x * gridDim.x * entries_per_thread) {
        for (int j = j_; j < min(samples, j_ + entries_per_thread); j++) {
            const int p = j * sample_interval;

            float alpha = 0.1;

            int i = d_C[p];
            const int c = d_C_sel[p];

            const float x = d_points[p * d + range_x];
            const float y = d_points[p * d + range_y];

            const bool hide = (selected_c != -2 && c != selected_c) ||
                              (range_x_min != range_x_max && range_y_min != range_y_max &&
                               !(range_x_min <= x && x <= range_x_max && range_y_min <= y && y <= range_y_max));

            if (hide) {
                alpha = 0.01;
            }

            if (i < 0 || !(d_D[i * d + x_dim] && d_D[i * d + y_dim])) {
                i = -1;
            }
            i++;

//            d_colors[j * color_components + 0] = colors[(i + 1) * 3 + 0];
//            d_colors[j * color_components + 1] = colors[(i + 1) * 3 + 1];
//            d_colors[j * color_components + 2] = colors[(i + 1) * 3 + 2];
//            d_colors[j * color_components + 3] = alpha;

            float r = 0.;
            float g = 0.;
            float b = 0.;


            int number_of_base_colors = 5;
            int number_of_colors = k;
            int number_of_shades = number_of_colors / number_of_base_colors;
            if (number_of_colors % number_of_base_colors) number_of_shades++;
            int shade = (i - 1) / number_of_base_colors;
            float shade_ration = shade / (number_of_shades - 1.);
            int base_color = ((i - 1) % number_of_base_colors) + 1;
            if (i == 0) {
                r = colors[0];
                g = colors[1];
                b = colors[2];
            } else if (number_of_shades > 1) {
                r = shade_ration * (colors[(5 + base_color) * 3 + 0] - colors[base_color * 3 + 0]) +
                    colors[base_color * 3 + 0];
                g = shade_ration * (colors[(5 + base_color) * 3 + 1] - colors[base_color * 3 + 1]) +
                    colors[base_color * 3 + 1];
                b = shade_ration * (colors[(5 + base_color) * 3 + 2] - colors[base_color * 3 + 2]) +
                    colors[base_color * 3 + 2];
            } else {
                r = (colors[(5 + base_color) * 3 + 0] + colors[base_color * 3 + 0]) / 2;
                g = (colors[(5 + base_color) * 3 + 1] + colors[base_color * 3 + 1]) / 2;
                b = (colors[(5 + base_color) * 3 + 2] + colors[base_color * 3 + 2]) / 2;
            }


            d_colors[j * color_components + 0] = r / 256.;
            d_colors[j * color_components + 1] = g / 256.;
            d_colors[j * color_components + 2] = b / 256.;
            d_colors[j * color_components + 3] = alpha;
        }
    }
};


__global__
void
kernel_scale_color(float *__restrict__ d_points_out, float *__restrict__ d_colors,
                   const float *__restrict__ d_points,
                   const int *__restrict__ d_C, const int *__restrict__ d_C_sel, const bool *__restrict__ d_D,
                   const int samples, const int d, const int k, const int x_dim, const int y_dim,
                   const int sample_interval, const int entries_per_thread,
                   const int selected_c, const float range_x_min, const float range_x_max, const float range_y_min,
                   const float range_y_max, const int range_x, const int range_y,
                   GLint x_offset, GLint y_offset,
                   float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                   float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {

    int color_components = 4;

    const float colors[13 * 3] = {
            62. / 256., 62. / 256., 62. / 256.,
            31. / 256., 120. / 256., 180. / 256.,
            51. / 256., 160. / 256., 44. / 256.,
            227. / 256., 26. / 256., 28. / 256.,
            255. / 256., 127. / 256., 0. / 256.,
            106. / 256., 61. / 256., 154. / 256.,
            177. / 256., 89. / 256., 40. / 256.,
            166. / 256., 206. / 256., 227. / 256.,
            178. / 256., 223. / 256., 138. / 256.,
            251. / 256., 154. / 256., 153. / 256.,
            253. / 256., 191. / 256., 111. / 256.,
            202. / 256., 178. / 256., 214. / 256.,
            255. / 256., 255. / 256., 153. / 256.
    };

    for (int j_ = (threadIdx.x + blockDim.x * blockIdx.x) * entries_per_thread;
         j_ < samples; j_ += blockDim.x * gridDim.x * entries_per_thread) {
        for (int j = j_; j < min(samples, j_ + entries_per_thread); j++) {
            const int p = j * sample_interval;

            float alpha = 0.1;

            int i = d_C[p];
            const int c = d_C_sel[p];

            const float x = d_points[p * d + range_x];
            const float y = d_points[p * d + range_y];

            const bool hide = (selected_c != -2 && c != selected_c) ||
                              (range_x_min != range_x_max && range_y_min != range_y_max &&
                               !(range_x_min <= x && x <= range_x_max && range_y_min <= y && y <= range_y_max));

            if (hide) {
                alpha = 0.01;
            }

            if (i < 0 || !(d_D[i * d + x_dim] && d_D[i * d + y_dim])) {
                i = -1;
            }


            d_points_out[j * 2 + 0] = x_offset + x_range_min +
                                      (x_range_max - x_range_min) * (d_points[p * d + x_dim] - x_domain_min) /
                                      (x_domain_max - x_domain_min);
            d_points_out[j * 2 + 1] = y_offset + y_range_min +
                                      (y_range_max - y_range_min) * (d_points[p * d + y_dim] - y_domain_min) /
                                      (y_domain_max - y_domain_min);

            d_colors[j * color_components + 0] = colors[(i + 1) * 3 + 0];
            d_colors[j * color_components + 1] = colors[(i + 1) * 3 + 1];
            d_colors[j * color_components + 2] = colors[(i + 1) * 3 + 2];
            d_colors[j * color_components + 3] = alpha;
        }
    }
};

/*
class Content : public Element {
protected:

public:
};
*/
//template<class T1>
__global__ // nvstd::function<int(T1*, int d, int p, int x_dim, int y_dim)> transform,
//void kernel_transform(T1 *d_data, float *d_out, int n, int d, int x_dim, int y_dim) {
void kernel_transform(int *d_data, float *d_out, int n, int d, int x_dim, int y_dim) {
    for (int p = threadIdx.x + blockDim.x * blockIdx.x; p < n; p += blockDim.x * gridDim.x) {
        int idx = d_data[p] + 1;// transform(d_data, d, p, x_dim, y_dim);
        //d_out[idx * 4 + 0] = idx;
        atomicAdd(&d_out[idx], 1.);
        //d_out[idx * 4 + 2] = idx;
        //d_out[idx * 4 + 3] = 0.;
    }
}

//template<class T1>
class Bars : public Element {
private:
    Scale *x_scale;
    Scale *y_scale;
    int count = 0;
    int spacing = 0;

    int n = 0;
    int d = 0;
    //std::function<T1* ()> get_data = [&]() {return nullptr;};
    std::function<int *()> get_data = [&]() { return nullptr; };

    int x_dim = 0;
    int y_dim = 0;

    //nvstd::function<int(T1*, int d, int p, int x_dim, int y_dim)> transform = nullptr; //[&] __device__(T1* d_Data, int d, int p)->T2 { return NULL; };
    nvstd::function<int(int *, int d, int p, int x_dim,
                        int y_dim)> transform = nullptr; //[&] __device__(T1* d_Data, int d, int p)->T2 { return NULL; };

    float *h_out = nullptr;

    std::vector <std::function<void(int) >> bar_click_listeners;

public:

    Bars(Scale *x_scale, Scale *y_scale) : x_scale(x_scale), y_scale(y_scale) {}

    void set_count(int c) {
        count = c;
        if (h_out != nullptr) {
            delete h_out;
        }
        h_out = new float[count];
    }

    void set_spacing(int s) {
        spacing = s;
    }

    //void set_data(std::function<T1* ()> f_data, int new_n, int new_d) {
    void set_data(std::function<int *()> f_data, int new_n, int new_d) {
        get_data = f_data;
        n = new_n;
        d = new_d;


    }

    //void set_transform(nvstd::function<int (T1*, int d, int p, int x_dim, int y_dim)> t) {
    void set_transform(nvstd::function<int(int *, int d, int p, int x_dim, int y_dim)> t) {
        transform = t;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x_scale->get_range_max() - x_scale->get_range_min();
        height = y_scale->get_range_max() - y_scale->get_range_min();
    }

    void paint() {
        /*cudaGraphicsMapResources(1, &(context.cuBuffer), 0);
        size_t num_bytes;
        float* cuVertex;
        cudaGraphicsResourceGetMappedPointer
        ((void**)&cuVertex, &num_bytes, context.cuBuffer);
        */

        float *d_out;
        cudaMalloc(&d_out, count * sizeof(float));
        cudaMemset(d_out, 0, count * sizeof(float));

        int *d_data = get_data();

        int number_of_blocks = n / 1024;
        if (n % number_of_blocks) number_of_blocks++;
        kernel_transform << < number_of_blocks, min(n, 1024) >> > (d_data, d_out, n, d, x_dim, y_dim);


        //number_of_blocks = count / 1024;
        //if (count % number_of_blocks) number_of_blocks++;
        /*kernel_scale << <1, min(count * 2, 1024) >> > (d_out, d_out, count * 2, 2, 0, 1,
            x_offset, y_offset,
            x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
            y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());*/


        cudaMemcpy(h_out, d_out, count * sizeof(float), cudaMemcpyDeviceToHost);

        /*
        glBegin(GL_LINES);
        glLineWidth(5);
        glColor3f(0.,0.,0.);
        for (int i = 0;i < count;i++) {
            for (int j = 0; j < 2;j++) {
                glVertex2i(h_out[i*4+j*2+0],h_out[i * 4 + j * 2 + 1]);
            }
        }
        glEnd();*/



        glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides

        float width = (x_scale->get_range_max() - x_scale->get_range_min() - (count + 1) * spacing) / count;
        float start = spacing;
        float end = spacing + width;

        int number_of_base_colors = 5;
        int number_of_colors = count - 1;
        int number_of_shades = number_of_colors / number_of_base_colors;
        if (number_of_colors % number_of_base_colors) number_of_shades++;


        for (int i = 0; i < count; i++) {
//            glColor3f(context.colors[i * 3 + 0] / 255., context.colors[i * 3 + 1] / 255.,
//                      context.colors[i * 3 + 2] / 255.);//Set the colour to red


            if (i == 0) {
                glColor3f(context.colors[i * 3 + 0] / 255., context.colors[i * 3 + 1] / 255.,
                          context.colors[i * 3 + 2] / 255.);//Set the colour to red
            } else {
                int shade = (i - 1) / number_of_base_colors;
                float shade_ration = shade / (number_of_shades - 1.);
                int base_color = ((i - 1) % number_of_base_colors) + 1;

                if (number_of_shades > 1) {

                    float r = shade_ration *
                              (context.colors[(5 + base_color) * 3 + 0] - context.colors[base_color * 3 + 0]) +
                              context.colors[base_color * 3 + 0];
                    float g = shade_ration *
                              (context.colors[(5 + base_color) * 3 + 1] - context.colors[base_color * 3 + 1]) +
                              context.colors[base_color * 3 + 1];
                    float b = shade_ration *
                              (context.colors[(5 + base_color) * 3 + 2] - context.colors[base_color * 3 + 2]) +
                              context.colors[base_color * 3 + 2];
                    glColor3f(r / 255., g / 255.,
                              b / 255.);//Set the colour to red
                } else {
                    float r = (context.colors[(5 + base_color) * 3 + 0] + context.colors[base_color * 3 + 0]) / 2;
                    float g = (context.colors[(5 + base_color) * 3 + 1] + context.colors[base_color * 3 + 1]) / 2;
                    float b = (context.colors[(5 + base_color) * 3 + 2] + context.colors[base_color * 3 + 2]) / 2;
                    glColor3f(r / 255., g / 255.,
                              b / 255.);//Set the colour to red
                }
            }


            float height = y_scale->get_range_min() + (y_scale->get_range_max() - y_scale->get_range_min()) *
                                                      (h_out[i] - y_scale->get_domain_min()) /
                                                      (y_scale->get_domain_max() - y_scale->get_domain_min());
            glVertex2i(x_offset + start, y_offset);            //Draw the four corners of the rectangle
            glVertex2i(x_offset + start, y_offset + height);
            glVertex2i(x_offset + end, y_offset + height);
            glVertex2i(x_offset + end, y_offset);


            if (x_offset + start <= context.mouse_x && context.mouse_x <= this->x_offset + end &&
                y_offset <= context.mouse_y && context.mouse_y <= this->y_offset + this->height) {


                int cluster_size = h_out[i];
                context.tooltip = [&, cluster_size]() {

                    int tooltip_margin = 8;
                    int tooltip_height = 24;
                    int tooltip_width = 100;


                    Text text_size(tooltip_width - 2 * tooltip_margin, tooltip_height);
                    text_size.set_text([&]() {
                        std::string str = "Size:" + std::to_string(cluster_size);
                        char *cstr = new char[str.length() + 1];
                        strcpy(cstr, str.c_str());
                        return cstr;
                    });
                    text_size.r = 1.;
                    text_size.g = 1.;
                    text_size.b = 1.;
                    text_size.set_font(GLUT_BITMAP_HELVETICA_12);
                    text_size.v_align_center();
                    text_size.h_align_left();
                    text_size.build(context.mouse_x + tooltip_margin, context.mouse_y);

                    tooltip_width = text_size.get_text_with() + 2 * tooltip_margin;

                    glBegin(GL_QUADS);
                    glColor3f(.3, .3, .3); //Set the colour to red
                    glVertex2i(context.mouse_x, context.mouse_y);
                    glVertex2i(context.mouse_x, context.mouse_y + tooltip_height);
                    glVertex2i(context.mouse_x + tooltip_width, context.mouse_y + tooltip_height);
                    glVertex2i(context.mouse_x + tooltip_width, context.mouse_y);
                    glEnd();

//                    glBegin(GL_LINES);
//                    glColor3f(.0, .0, .0);
//                    glVertex2i(context.mouse_x, context.mouse_y);
//                    glVertex2i(context.mouse_x, context.mouse_y + 20);
//
//                    glVertex2i(context.mouse_x, context.mouse_y + 20);
//                    glVertex2i(context.mouse_x + 100, context.mouse_y + 20);
//
//                    glVertex2i(context.mouse_x + 100, context.mouse_y + 20);
//                    glVertex2i(context.mouse_x + 100, context.mouse_y);
//
//                    glVertex2i(context.mouse_x + 100, context.mouse_y);
//                    glVertex2i(context.mouse_x, context.mouse_y);
//                    glEnd();

                    text_size.paint();

                };
            }


            start += spacing + width;
            end += spacing + width;
        }
        glEnd();


//        printf("x: %d, %d, %d\n", x_offset, context.mouse_x, this->width);
//        printf("y: %d, %d, %d\n", y_offset, context.mouse_y, height);
//
//        if (this->x_offset <= context.mouse_x && context.mouse_x <= this->x_offset + this->width &&
//            this->y_offset <= context.mouse_y && context.mouse_y <= this->y_offset + this->height) {
//            printf("test\n");
//            glBegin(GL_LINES);
//            glColor3f(.0, .0, .0);
//            glVertex2i(x_offset, y_offset);
//            glVertex2i(x_offset, y_offset + this->height);
//
//            glVertex2i(x_offset, y_offset + this->height);
//            glVertex2i(x_offset + this->width, y_offset + this->height);
//
//            glVertex2i(x_offset + this->width, y_offset + this->height);
//            glVertex2i(x_offset + this->width, y_offset);
//            glEnd();
//        }

        cudaFree(d_out);
    }

    void on_mouse_changed(int x, int y, int state) {

        if (state == MOUSE_UP) {
            float width = (x_scale->get_range_max() - x_scale->get_range_min() - (count + 1) * spacing) / count;
            float start = spacing;
            float end = spacing + width;

            for (int i = 0; i < count; i++) {

                float height = y_scale->get_range_min() + (y_scale->get_range_max() - y_scale->get_range_min()) *
                                                          (h_out[i] - y_scale->get_domain_min()) /
                                                          (y_scale->get_domain_max() - y_scale->get_domain_min());

                if (x_offset + start <= x && x <= x_offset + end && y_offset <= y && y <= y_offset + height) {
                    for (int j = 0; j < bar_click_listeners.size(); j++) {
                        std::function<void(int)> l = bar_click_listeners[j];
                        l(i);
                    }
                }

                start += spacing + width;
                end += spacing + width;
            }
        }
    }

    void add_bar_click_listener(std::function<void(int)> l) {
        bar_click_listeners.push_back(l);
    }
};


class Points : public Element {
private:
    Scale *x_scale;
    Scale *y_scale;
    int sample_interval = 1;

    float *d_points;
    float *h_points;
    int n;
    int d;

    int x_dim = 0;
    int y_dim = 1;

    int size = 1;

    std::function<int()> get_k = [&]() { return context.k; };
    std::function<int()> get_l = [&]() { return context.l; };
public:
    Points(Scale *x_scale, Scale *y_scale, float *d_points, float *h_points, int n, int d) : x_scale(x_scale),
                                                                                             y_scale(y_scale),
                                                                                             d_points(d_points),
                                                                                             h_points(h_points),
                                                                                             n(n),
                                                                                             d(d) {}

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_size(int s) {
        size = s;
    }

    void set_sampling_interval(int s) {
        sample_interval = s;
    }

    void set_dims(int new_x_dim, int new_y_dim) {
        x_dim = new_x_dim;
        y_dim = new_y_dim;
    }

    void set_k(std::function<int()> f_k) {
        get_k = f_k;
    }

    void set_l(std::function<int()> f_l) {
        get_l = f_l;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x_scale->get_range_max() - x_scale->get_range_min();
        height = y_scale->get_range_max() - y_scale->get_range_min();
    }

    void paint() {

        int samples = n / sample_interval;
        glEnable(GL_POINT_SMOOTH);

        if (context.use_GPU) {
            cudaGraphicsMapResources(1, &(context.cuBuffer), 0);
            size_t num_bytes;
            float *cuVertex;
            cudaGraphicsResourceGetMappedPointer
                    ((void **) &cuVertex, &num_bytes, context.cuBuffer);

            int block_size = 32;
            int entries_per_thread = 2;

            int number_of_blocks = samples / (block_size * entries_per_thread);
            if (samples % (block_size * entries_per_thread)) number_of_blocks++;


            kernel_scale << < number_of_blocks, min(samples,
                                                    block_size) >> > (cuVertex, d_points, samples, d, x_dim, y_dim, sample_interval, entries_per_thread,
                    x_offset, y_offset,
                    x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                    y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());


            float *d_colors = &cuVertex[samples * 2];
            entries_per_thread = 8;
            number_of_blocks = samples / (block_size * entries_per_thread);
            if (samples % (block_size * entries_per_thread)) number_of_blocks++;

            kernel_color << < number_of_blocks, min(samples,
                                                    block_size) >> > (d_colors, d_points, context.get_result(
                    get_k(), get_l()).d_C, context.get_result(context.k, context.l).d_C, context.get_result(get_k(),
                                                                                                            get_l()).d_D, samples, d, get_k(), x_dim, y_dim, sample_interval, entries_per_thread,
                    context.selected_c, context.range_x_min, context.range_x_max, context.range_y_min, context.range_y_max, context.range_x, context.range_y);

            /*kernel_scale_color << <number_of_blocks, min(samples, block_size) >> > (cuVertex, d_colors, d_points, context.get_result(get_k(), get_l()).d_C, context.get_result(context.k, context.l).d_C, context.get_result(get_k(), get_l()).d_D, samples, d, get_k(), x_dim, y_dim, sample_interval, entries_per_thread,
                context.selected_c, context.range_x_min, context.range_x_max, context.range_y_min, context.range_y_max, context.range_x, context.range_y,
                x_offset, y_offset,
                x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());*/

            // unmap buffer object
            cudaGraphicsUnmapResources(1, &(context.cuBuffer));


            // render from the points
            glBindBuffer(GL_ARRAY_BUFFER, context.glBuffer);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glVertexPointer(2, GL_FLOAT, 0, 0);

            //glColor3f(context.t, 1. - context.t, 1. - context.t);
            glColorPointer(4, GL_FLOAT, 0, (char *) 0 + samples * 2 * sizeof(float));

            glPointSize(size);
            glDrawArrays(GL_POINTS, 0, samples);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
            glFlush();

        } else {

            float colors[13 * 3] = {
                    62, 62, 62,
                    31, 120, 180,
                    51, 160, 44,
                    227, 26, 28,
                    255, 127, 0,
                    106, 61, 154,
                    177, 89, 40,
                    166, 206, 227,
                    178, 223, 138,
                    251, 154, 153,
                    253, 191, 111,
                    202, 178, 214,
                    255, 255, 153
            };


            int *h_C = context.get_result(get_k(), get_l()).get_h_C();
            int *h_C_sel = context.get_result(context.k, context.l).get_h_C();
            bool *h_D = context.get_result(get_k(), get_l()).get_h_D();

            glEnable(GL_PROGRAM_POINT_SIZE);
            glPointSize(size);
            glBegin(GL_POINTS);
            for (int j = 0; j < samples; j++) {
                int p = j * sample_interval;

                // scale points
                float x_scaled = x_offset + x_scale->get_range_min() +
                                 (x_scale->get_range_max() - x_scale->get_range_min()) *
                                 (h_points[p * d + x_dim] - x_scale->get_domain_min()) /
                                 (x_scale->get_domain_max() - x_scale->get_domain_min());
                float y_scaled = y_offset + y_scale->get_range_min() +
                                 (y_scale->get_range_max() - y_scale->get_range_min()) *
                                 (h_points[p * d + y_dim] - y_scale->get_domain_min()) /
                                 (y_scale->get_domain_max() - y_scale->get_domain_min());

                float alpha = 0.1;

                int i = h_C[p];
                int c = h_C_sel[p];

                if (context.selected_c != -2 && c != context.selected_c) {
                    alpha = 0.01;
                }

                if (context.range_x_min != context.range_x_max && context.range_y_min != context.range_y_max) {

                    float x = h_points[p * d + context.range_x];
                    float y = h_points[p * d + context.range_y];
                    if (context.range_x_min <= x && x <= context.range_x_max && context.range_y_min <= y &&
                        y <= context.range_y_max) {

                    } else {
                        alpha = 0.01;
                    }
                }

                if (i < 0 || !(h_D[i * d + x_dim] && h_D[i * d + y_dim])) {
                    i = -1;
                }

                float r = colors[(i + 1) * 3 + 0] / 255.;
                float g = colors[(i + 1) * 3 + 1] / 255.;
                float b = colors[(i + 1) * 3 + 2] / 255.;

                //glClear(GL_COLOR_BUFFER_BIT);
                glColor4f(r, g, b, alpha);
                glVertex2f(x_scaled, y_scaled);
            }
            glEnd();
            glFlush();

        }
    }
};


__global__
void kernel_heatmap(float *cuVertex, int x_offset, int y_offset, int width, int height,
                    float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                    float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float tile_x_width = ((float) x_range_max - x_range_min) / (float) width;
    float tile_y_width = ((float) y_range_max - y_range_min) / (float) height;

    int offset = x * height * 8 + y * 8;

    cuVertex[offset + 0] = x_offset + x * tile_x_width;
    cuVertex[offset + 1] = y_offset + y * tile_y_width;

    cuVertex[offset + 2] = x_offset + x * tile_x_width;
    cuVertex[offset + 3] = y_offset + (y + 1) * tile_y_width;

    cuVertex[offset + 4] = x_offset + (x + 1) * tile_x_width;
    cuVertex[offset + 5] = y_offset + (y + 1) * tile_y_width;

    cuVertex[offset + 6] = x_offset + (x + 1) * tile_x_width;
    cuVertex[offset + 7] = y_offset + y * tile_y_width;

}


__global__
void kernel_heatmap_color(float *d_colors, float *d_points, int n, int d, int x_dim, int y_dim,
                          int *d_C_sel, int selected_c,
                          int width, int height, int entries_per_thread,
                          float range_x_min, float range_x_max, float range_y_min, float range_y_max, int range_x,
                          int range_y,
                          float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                          float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {

    int color_components = 4;

    float tile_x_width = ((float) x_range_max - x_range_min) / (float) width;
    float tile_y_width = ((float) y_range_max - y_range_min) / (float) height;

    for (int j_ = (threadIdx.x + blockDim.x * blockIdx.x) * entries_per_thread;
         j_ < n; j_ += blockDim.x * gridDim.x * entries_per_thread) {
        for (int j = j_; j < min(n, j_ + entries_per_thread); j++) {

            int c = d_C_sel[j];


            int x = (x_range_min + (x_range_max - x_range_min) * (d_points[j * d + x_dim] - x_domain_min) /
                                   (x_domain_max - x_domain_min)) / tile_x_width;
            int y = (y_range_min + (y_range_max - y_range_min) * (d_points[j * d + y_dim] - y_domain_min) /
                                   (y_domain_max - y_domain_min)) / tile_y_width;

            int idx = x * height + y;

            if (selected_c == -2 || c == selected_c) {
                if (range_x_min != range_x_max && range_y_min != range_y_max) {
                    float x = d_points[j * d + range_x];
                    float y = d_points[j * d + range_y];
                    if (range_x_min <= x && x <= range_x_max && range_y_min <= y && y <= range_y_max) {
                        for (int i = 0; i < 4; i++) {
                            atomicAdd(&d_colors[idx * 4 * color_components + i * color_components + 3],
                                      ((float) width) / n);
                        }
                    }
                } else {
                    for (int i = 0; i < 4; i++) {
                        atomicAdd(&d_colors[idx * 4 * color_components + i * color_components + 3],
                                  ((float) width) / n);
                    }
                }
            }
        }
    }
};


__global__
void kernel_heatmap_color_3(float *d_colors, float *d_points, int n, int d, int x_dim, int y_dim,
                            int *d_C_sel, int selected_c,
                            int width, int height, int entries_per_thread,
                            float range_x_min, float range_x_max, float range_y_min, float range_y_max, int range_x,
                            int range_y,
                            float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                            float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {

    extern __shared__ float s_colors[];


    for (int j = threadIdx.x; j < height * width; j += blockDim.x) {
        s_colors[j] = 0.;
    }
    __syncthreads();

    int color_components = 4;

    float tile_x_width = ((float) x_range_max - x_range_min) / (float) width;
    float tile_y_width = ((float) y_range_max - y_range_min) / (float) height;

    for (int j_ = (threadIdx.x + blockDim.x * blockIdx.x) * entries_per_thread;
         j_ < n; j_ += blockDim.x * gridDim.x * entries_per_thread) {
        for (int j = j_; j < min(n, j_ + entries_per_thread); j++) {

            int c = d_C_sel[j];


            int x = (x_range_min + (x_range_max - x_range_min) * (d_points[j * d + x_dim] - x_domain_min) /
                                   (x_domain_max - x_domain_min)) / tile_x_width;
            if (x == width) x--;
            int y = (y_range_min + (y_range_max - y_range_min) * (d_points[j * d + y_dim] - y_domain_min) /
                                   (y_domain_max - y_domain_min)) / tile_y_width;
            if (y == height)y--;

            int idx = x * height + y;

            if (selected_c == -2 || c == selected_c) {
                if (range_x_min != range_x_max && range_y_min != range_y_max) {
                    float x = d_points[j * d + range_x];
                    float y = d_points[j * d + range_y];
                    if (range_x_min <= x && x <= range_x_max && range_y_min <= y && y <= range_y_max) {
                        atomicAdd(&s_colors[idx], ((float) width) / n);
                    }
                } else {
                    atomicAdd(&s_colors[idx], ((float) width) / n);
                }
            }
        }
    }

    __syncthreads();

    for (int j = threadIdx.x; j < height * width; j += blockDim.x) {
        for (int i = 0; i < 4; i++) {
            atomicAdd(&d_colors[j * 4 * color_components + i * color_components + 3], s_colors[j]);
        }
    }
};


class Heatmap : Element {
private:
    Scale *x_scale;
    Scale *y_scale;

    int n;
    int d;
    float *d_points;
    float *h_points;

    int x_dim = 0;
    int y_dim = 1;

    int size = 1;

    std::function<int()> get_k = [&]() { return context.k; };
    std::function<int()> get_l = [&]() { return context.l; };

public:

    Heatmap(Scale *x_scale, Scale *y_scale, float *d_points, float *h_points, int n, int d) : x_scale(x_scale),
                                                                                              y_scale(y_scale),
                                                                                              d_points(d_points),
                                                                                              h_points(h_points),
                                                                                              n(n),
                                                                                              d(d) {}

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_size(int s) {
        size = s;
    }

    void set_dims(int new_x_dim, int new_y_dim) {
        x_dim = new_x_dim;
        y_dim = new_y_dim;
    }

    void set_k(std::function<int()> f_k) {
        get_k = f_k;
    }

    void set_l(std::function<int()> f_l) {
        get_l = f_l;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x_scale->get_range_max() - x_scale->get_range_min();
        height = y_scale->get_range_max() - y_scale->get_range_min();
    }

    void paint() {

        if (context.use_GPU) {

            cudaGraphicsMapResources(1, &(context.cuBuffer), 0);
            size_t num_bytes;
            float *cuVertex;
            cudaGraphicsResourceGetMappedPointer
                    ((void **) &cuVertex, &num_bytes, context.cuBuffer);

            dim3 block(32, 32);

            kernel_heatmap << < 1, block >> > (cuVertex, x_offset, y_offset, 32, 32,
                    x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                    y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());


            float *d_colors = &cuVertex[32 * 32 * 2 * 4];

            cudaMemset(d_colors, 0, 32 * 32 * 4 * 4 * sizeof(float));

            int block_size = 1024;

            int entries_per_thread = 64;
            int number_of_blocks = n / (block_size * entries_per_thread);
            if (n % (block_size * entries_per_thread)) number_of_blocks++;

            kernel_heatmap_color_3 << < number_of_blocks, min(n, block_size), 32 * 32 *
                                                                              sizeof(float) >> > (d_colors, d_points, n, d, x_dim, y_dim, context.get_result(
                    context.k, context.l).d_C, context.selected_c,
                    32, 32, entries_per_thread,
                    context.range_x_min, context.range_x_max, context.range_y_min, context.range_y_max, context.range_x, context.range_y,
                    x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                    y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());

            // unmap buffer object
            cudaGraphicsUnmapResources(1, &(context.cuBuffer));


            // render from the points
            glBindBuffer(GL_ARRAY_BUFFER, context.glBuffer);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glVertexPointer(2, GL_FLOAT, 0, 0);

            //glColor3f(context.t, 1. - context.t, 1. - context.t);
            glColorPointer(4, GL_FLOAT, 0, (char *) 0 + 32 * 32 * 2 * 4 * sizeof(float));

            glPointSize(size);
            glDrawArrays(GL_QUADS, 0, 4 * 32 * 32);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
        } else {


            int *h_C = context.get_result(get_k(), get_l()).get_h_C();
            int *h_C_sel = context.get_result(context.k, context.l).get_h_C();
            bool *h_D = context.get_result(get_k(), get_l()).get_h_D();


            float tile_x_width = (x_scale->get_range_max() - x_scale->get_range_min()) / 32;
            float tile_y_width = (y_scale->get_range_max() - y_scale->get_range_min()) / 32;

            float *h_colors = new float[32 * 32];
            for (int i = 0; i < 32 * 32; i++) {
                h_colors[i] = 0.;
            }

            for (int p = 0; p < n; p++) {
                int c = h_C_sel[p];

                int x = (x_scale->get_range_min() + (x_scale->get_range_max() - x_scale->get_range_min()) *
                                                    (h_points[p * d + x_dim] - x_scale->get_domain_min()) /
                                                    (x_scale->get_domain_max() - x_scale->get_domain_min())) /
                        tile_x_width;
                int y = (y_scale->get_range_min() + (y_scale->get_range_max() - y_scale->get_range_min()) *
                                                    (h_points[p * d + y_dim] - y_scale->get_domain_min()) /
                                                    (y_scale->get_domain_max() - y_scale->get_domain_min())) /
                        tile_y_width;

                int idx = x * 32 + y;

                if (context.selected_c == -2 || c == context.selected_c) {
                    if (context.range_x_min != context.range_x_max && context.range_y_min != context.range_y_max) {
                        float x = h_points[p * d + context.range_x];
                        float y = h_points[p * d + context.range_y];
                        if (context.range_x_min <= x && x <= context.range_x_max && context.range_y_min <= y &&
                            y <= context.range_y_max) {
                            h_colors[idx] += 32. / n;
                        }
                    } else {
                        h_colors[idx] += 32. / n;
                    }
                }
            }

            glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides


            for (int x = 0; x < 32; x++) {
                for (int y = 0; y < 32; y++) {

                    int idx = x * 32 + y;

                    glColor4f(0., 0., 0., h_colors[idx]);
                    glVertex2i(x_offset + x * tile_x_width,
                               y_offset + y * tile_y_width);            //Draw the four corners of the rectangle
                    glVertex2i(x_offset + x * tile_x_width, y_offset + (y + 1) * tile_y_width);
                    glVertex2i(x_offset + (x + 1) * tile_x_width, y_offset + (y + 1) * tile_y_width);
                    glVertex2i(x_offset + (x + 1) * tile_x_width, y_offset + y * tile_y_width);
                }
            }

            glEnd();
            glFlush();
            delete h_colors;
        }
    }
};


__global__
void kernel_small_points(float *cuVertex, int x_offset, int y_offset, int width, int height,
                         float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                         float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width)
        return;
    if (y >= height)
        return;

    int offset = x * height * 2 + y * 2;

    cuVertex[offset + 0] = x_offset + x;
    cuVertex[offset + 1] = y_offset + y;

    offset = width * height * 2 + x * height * 3 + y * 3;
    cuVertex[offset + 0] = 1.;
    cuVertex[offset + 1] = 1.;
    cuVertex[offset + 2] = 1.;

}

__global__
void kernel_small_points_color(float *d_colors, float *d_points, int n, int d, int x_dim, int y_dim,
                               int *d_C, int *d_C_sel, bool *d_D, int selected_c,
                               int width, int height,
                               float range_x_min, float range_x_max, float range_y_min, float range_y_max,
                               int range_x,
                               int range_y,
                               float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
                               float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {

    int color_components = 3;

    float colors[13 * 3] = {
            62 / 255., 62 / 255., 62 / 255.,
            31 / 255., 120 / 255., 180 / 255.,
            51 / 255., 160 / 255., 44 / 255.,
            227 / 255., 26 / 255., 28 / 255.,
            255 / 255., 127 / 255., 0 / 255.,
            106 / 255., 61 / 255., 154 / 255.,
            177 / 255., 89 / 255., 40 / 255.,
            166 / 255., 206 / 255., 227 / 255.,
            178 / 255., 223 / 255., 138 / 255.,
            251 / 255., 154 / 255., 153 / 255.,
            253 / 255., 191 / 255., 111 / 255.,
            202 / 255., 178 / 255., 214 / 255.,
            255 / 255., 255 / 255., 153 / 255.
    };

    for (int j = threadIdx.x + blockDim.x * blockIdx.x; j < n; j += blockDim.x * gridDim.x) {
        int p = j;
        int i = d_C[j];
        int c = d_C_sel[j];


        int px_x = (x_range_min + (x_range_max - x_range_min) * (d_points[j * d + x_dim] - x_domain_min) /
                                  (x_domain_max - x_domain_min));
        int px_y = (y_range_min + (y_range_max - y_range_min) * (d_points[j * d + y_dim] - y_domain_min) /
                                  (y_domain_max - y_domain_min));

        int idx = px_x * height + px_y;

        float alpha = 0.1;

        float x = d_points[p * d + range_x];
        float y = d_points[p * d + range_y];

        bool hide = (selected_c != -2 && c != selected_c) ||
                    (range_x_min != range_x_max && range_y_min != range_y_max &&
                     !(range_x_min <= x && x <= range_x_max && range_y_min <= y && y <= range_y_max));

        if (hide) {
            alpha = 0.01;
        }

        if (i < 0 || !(d_D[i * d + x_dim] && d_D[i * d + y_dim])) {
            i = -1;
        }
        float r = colors[(i + 1) * 3 + 0];
        float g = colors[(i + 1) * 3 + 1];
        float b = colors[(i + 1) * 3 + 2];

        //todo should be locked
        d_colors[idx * color_components + 0] = (1. - alpha) * d_colors[idx * color_components + 0] + alpha * r;
        d_colors[idx * color_components + 1] = (1. - alpha) * d_colors[idx * color_components + 1] + alpha * g;
        d_colors[idx * color_components + 2] = (1. - alpha) * d_colors[idx * color_components + 2] + alpha * b;
    }
};

class Small_points : Element {
private:
    Scale *x_scale;
    Scale *y_scale;

    int n;
    int d;
    float *d_points;
    float *h_points;

    int x_dim = 0;
    int y_dim = 1;

    int size = 1;

    std::function<int()> get_k = [&]() { return context.k; };
    std::function<int()> get_l = [&]() { return context.l; };

public:

    Small_points(Scale *x_scale, Scale *y_scale, float *d_points, float *h_points, int n, int d) : x_scale(x_scale),
                                                                                                   y_scale(y_scale),
                                                                                                   d_points(
                                                                                                           d_points),
                                                                                                   h_points(
                                                                                                           h_points),
                                                                                                   n(n), d(d) {}

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_size(int s) {
        size = s;
    }

    void set_dims(int new_x_dim, int new_y_dim) {
        x_dim = new_x_dim;
        y_dim = new_y_dim;
    }

    void set_k(std::function<int()> f_k) {
        get_k = f_k;
    }

    void set_l(std::function<int()> f_l) {
        get_l = f_l;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x_scale->get_range_max() - x_scale->get_range_min();
        height = y_scale->get_range_max() - y_scale->get_range_min();
    }

    void paint() {

        if (context.use_GPU) {


            width = x_scale->get_range_max() - x_scale->get_range_min();
            height = y_scale->get_range_max() - y_scale->get_range_min();

            cudaGraphicsMapResources(1, &(context.cuBuffer), 0);
            size_t num_bytes;
            float *cuVertex;
            cudaGraphicsResourceGetMappedPointer
                    ((void **) &cuVertex, &num_bytes, context.cuBuffer);


            int grid_1 = width / 32;
            if (width % 32) grid_1++;
            int grid_2 = height / 32;
            if (height % 32) grid_2++;
            dim3 grid(grid_1, grid_2);
            dim3 block(32, 32);

            kernel_small_points << < grid, block >> > (cuVertex, x_offset, y_offset, width, height,
                    x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                    y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());


            float *d_colors = &cuVertex[width * height * 2];

            //cudaMemset(d_colors, 0, width * height * 3 * sizeof(float));

            int number_of_blocks = n / 1024;
            if (n % number_of_blocks) number_of_blocks++;
            kernel_small_points_color << < number_of_blocks, min(n,
                                                                 1024) >> > (d_colors, d_points, n, d, x_dim, y_dim, context.get_result(
                    get_k(), get_l()).d_C, context.get_result(context.k, context.l).d_C, context.get_result(
                    context.k,
                    context.l).d_D, context.selected_c,
                    width, height,
                    context.range_x_min, context.range_x_max, context.range_y_min, context.range_y_max, context.range_x, context.range_y,
                    x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                    y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());

            // unmap buffer object
            cudaGraphicsUnmapResources(1, &(context.cuBuffer));


            // render from the points
            glBindBuffer(GL_ARRAY_BUFFER, context.glBuffer);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glVertexPointer(2, GL_FLOAT, 0, 0);

            //glColor3f(context.t, 1. - context.t, 1. - context.t);
            glColorPointer(3, GL_FLOAT, 0, (char *) 0 + width * height * 2 * sizeof(float));

            glPointSize(size);
            glDrawArrays(GL_POINTS, 0, width * height);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
        } else {


            int *h_C = context.get_result(get_k(), get_l()).get_h_C();
            int *h_C_sel = context.get_result(context.k, context.l).get_h_C();
            bool *h_D = context.get_result(get_k(), get_l()).get_h_D();


            float tile_x_width = (x_scale->get_range_max() - x_scale->get_range_min()) / 32;
            float tile_y_width = (y_scale->get_range_max() - y_scale->get_range_min()) / 32;

            float *h_colors = new float[32 * 32];
            for (int i = 0; i < 32 * 32; i++) {
                h_colors[i] = 0.;
            }

            for (int p = 0; p < n; p++) {
                int c = h_C_sel[p];

                int x = (x_scale->get_range_min() + (x_scale->get_range_max() - x_scale->get_range_min()) *
                                                    (h_points[p * d + x_dim] - x_scale->get_domain_min()) /
                                                    (x_scale->get_domain_max() - x_scale->get_domain_min())) /
                        tile_x_width;
                int y = (y_scale->get_range_min() + (y_scale->get_range_max() - y_scale->get_range_min()) *
                                                    (h_points[p * d + y_dim] - y_scale->get_domain_min()) /
                                                    (y_scale->get_domain_max() - y_scale->get_domain_min())) /
                        tile_y_width;

                int idx = x * 32 + y;

                if (context.selected_c == -2 || c == context.selected_c) {
                    if (context.range_x_min != context.range_x_max && context.range_y_min != context.range_y_max) {
                        float x = h_points[p * d + context.range_x];
                        float y = h_points[p * d + context.range_y];
                        if (context.range_x_min <= x && x <= context.range_x_max && context.range_y_min <= y &&
                            y <= context.range_y_max) {
                            h_colors[idx] += 32. / n;
                        }
                    } else {
                        h_colors[idx] += 32. / n;
                    }
                }
            }

            glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides


            for (int x = 0; x < 32; x++) {
                for (int y = 0; y < 32; y++) {

                    int idx = x * 32 + y;

                    glColor4f(0., 0., 0., h_colors[idx]);
                    glVertex2i(x_offset + x * tile_x_width,
                               y_offset + y * tile_y_width);            //Draw the four corners of the rectangle
                    glVertex2i(x_offset + x * tile_x_width, y_offset + (y + 1) * tile_y_width);
                    glVertex2i(x_offset + (x + 1) * tile_x_width, y_offset + (y + 1) * tile_y_width);
                    glVertex2i(x_offset + (x + 1) * tile_x_width, y_offset + y * tile_y_width);
                }
            }

            glEnd();
            delete h_colors;
        }
    }
};


__device__
float K(float x) {
    float pi = 2 * acos(0.0);
    return pow(2. * pi, -1 / 2) * exp(-pow(x, 2) / 2);
}

__global__
void
kernel_density(float *d_density, float *d_points, int n, int d, int x_width, int y_width, int x_offset,
               int y_offset,
               int x_dim, int y_dim,
               float x_domain_min, float x_domain_max, float x_range_min, float x_range_max,
               float y_domain_min, float y_domain_max, float y_range_min, float y_range_max) {
    int px_x = blockIdx.x;
    int px_y = blockIdx.y;
    int px_idx = px_x * y_width + px_y;

    if (threadIdx.x == 0) {
        d_density[px_idx * 2 + 0] = x_offset + px_x;
        d_density[px_idx * 2 + 1] = y_offset + px_y;
    }

    float h = 2.5;
    float f = 0.;
    for (int p = threadIdx.x; p < n; p += blockDim.x) {

        float p_x = x_range_min + (x_range_max - x_range_min) * (d_points[p * d + x_dim] - x_domain_min) /
                                  (x_domain_max - x_domain_min);
        float p_y = y_range_min + (y_range_max - y_range_min) * (d_points[p * d + y_dim] - y_domain_min) /
                                  (y_domain_max - y_domain_min);

        float dist1 = px_x - p_x;
        dist1 *= dist1;
        float dist2 = px_y - p_y;
        dist2 *= dist2;

        float dist = sqrt(dist1 + dist2);

        f += 1. / (n * h) * K(dist / h);

    }
    atomicAdd(&d_density[x_width * y_width * 2 + px_idx * 4 + 3], f * 10);
}

class Density : Element {
private:
    Scale *x_scale;
    Scale *y_scale;

    int n;
    int d;
    float *d_points;
    float *d_density;

    int x_dim = 0;
    int y_dim = 1;

    int size = 1;

    std::function<int()> get_k = [&]() { return context.k; };
    std::function<int()> get_l = [&]() { return context.l; };

public:

    Density(Scale *x_scale, Scale *y_scale, float *d_points, int n, int d) : x_scale(x_scale), y_scale(y_scale),
                                                                             d_points(d_points), n(n), d(d) {

        int x_width = x_scale->get_range_max() - x_scale->get_range_min();
        int y_width = y_scale->get_range_max() - y_scale->get_range_min();

        int px_in_plot = x_width * y_width;

        cudaMalloc(&d_density, sizeof(float) * px_in_plot * 6);
        cudaMemset(d_density, 0, sizeof(float) * px_in_plot * 6);

    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_size(int s) {
        size = s;
    }

    void set_dims(int new_x_dim, int new_y_dim) {
        x_dim = new_x_dim;
        y_dim = new_y_dim;
    }

    void set_k(std::function<int()> f_k) {
        get_k = f_k;
    }

    void set_l(std::function<int()> f_l) {
        get_l = f_l;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
        width = x_scale->get_range_max() - x_scale->get_range_min();
        height = y_scale->get_range_max() - y_scale->get_range_min();

        dim3 grid(width, height);
        kernel_density << < grid, min(n,
                                      1024) >> > (d_density, d_points, n, d, width, height, x_offset, y_offset, x_dim, y_dim,
                x_scale->get_domain_min(), x_scale->get_domain_max(), x_scale->get_range_min(), x_scale->get_range_max(),
                y_scale->get_domain_min(), y_scale->get_domain_max(), y_scale->get_range_min(), y_scale->get_range_max());
    }

    void paint() {

        cudaGraphicsMapResources(1, &(context.cuBuffer), 0);
        size_t num_bytes;
        float *cuVertex;
        cudaGraphicsResourceGetMappedPointer
                ((void **) &cuVertex, &num_bytes, context.cuBuffer);

        cudaMemcpy(cuVertex, d_density, sizeof(float) * width * height * 6, cudaMemcpyDeviceToDevice);

        // unmap buffer object
        cudaGraphicsUnmapResources(1, &(context.cuBuffer));


        // render from the points
        glBindBuffer(GL_ARRAY_BUFFER, context.glBuffer);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glVertexPointer(2, GL_FLOAT, 0, 0);

        //glColor3f(context.t, 1. - context.t, 1. - context.t);
        glColorPointer(4, GL_FLOAT, 0, (char *) 0 + width * height * 2 * sizeof(float));

        glPointSize(size);
        glDrawArrays(GL_POINTS, 0, width * height);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
};


class Plot : Element {
private:
    Table *plt_table;
    Scale *x_scale;
    Scale *y_scale;

    Axis *x_axis;
    Axis *y_axis;

    int x_axis_height = 0;
    int y_axis_width = 0;
    /*int x_dim = 0;
    int y_dim = 1;

    std::function<int()> get_k = [&]() {return context.k;};
    std::function<int()> get_l = [&]() {return context.l;};*/
    std::function<bool()> get_selected = [&]() { return false; };

    Group *content_group;

    bool range_selection = false;

public:
    Plot(Scale *x_scale, Scale *y_scale) : x_scale(x_scale), y_scale(y_scale) {
        content_group = new Group();

        x_axis = new Axis(x_scale);
        y_axis = new Axis(y_scale);

        this->add_down_listener([&]() {
            if (range_selection) {
                if (!context.range_selection_started) {
                    context.range_x_start = context.mouse_x;
                    context.range_y_start = context.mouse_y;
                    context.range_selection_started = true;
                    context.range_x = context.selected_i;
                    context.range_y = context.selected_j;
                }
                context.range_x_end = context.mouse_x;
                context.range_y_end = context.mouse_y;

                this->convert_range();

            }
        });
        this->add_click_listener([&]() {
            if (range_selection) {
                context.range_selection_started = false;
                context.range_x_end = context.mouse_x;
                context.range_y_end = context.mouse_y;

                this->convert_range();
            }
        });
        this->add_move_listener([&]() {
            if (range_selection) {
                if (context.range_selection_started) {//todo not a good locations for this
                    context.range_x_end = context.mouse_x;
                    context.range_y_end = context.mouse_y;
                }
                this->convert_range();
            }
        });
    }

    ~Plot() {
        //delete plt_table;
    }

    Axis *get_x_axis() {
        return x_axis;
    }

    Axis *get_y_axis() {
        return y_axis;
    }

    void convert_range() {
        context.range_x_min =
                context.range_x_start < context.range_x_end ? context.range_x_start : context.range_x_end;
        context.range_x_max =
                context.range_x_start < context.range_x_end ? context.range_x_end : context.range_x_start;
        context.range_y_min =
                context.range_y_start < context.range_y_end ? context.range_y_start : context.range_y_end;
        context.range_y_max =
                context.range_y_start < context.range_y_end ? context.range_y_end : context.range_y_start;

        context.range_x_min -= x_offset + y_axis_width;
        context.range_x_max -= x_offset + y_axis_width;
        context.range_y_min -= y_offset + x_axis_height;
        context.range_y_max -= y_offset + x_axis_height;

        context.range_x_min = (context.range_x_min - x_scale->get_range_min()) /
                              (x_scale->get_range_max() - x_scale->get_range_min()) *
                              (x_scale->get_domain_max() - x_scale->get_domain_min()) + x_scale->get_domain_min();
        context.range_x_max = (context.range_x_max - x_scale->get_range_min()) /
                              (x_scale->get_range_max() - x_scale->get_range_min()) *
                              (x_scale->get_domain_max() - x_scale->get_domain_min()) + x_scale->get_domain_min();
        context.range_y_min = (context.range_y_min - y_scale->get_range_min()) /
                              (y_scale->get_range_max() - y_scale->get_range_min()) *
                              (y_scale->get_domain_max() - y_scale->get_domain_min()) + y_scale->get_domain_min();
        context.range_y_max = (context.range_y_max - y_scale->get_range_min()) /
                              (y_scale->get_range_max() - y_scale->get_range_min()) *
                              (y_scale->get_domain_max() - y_scale->get_domain_min()) + y_scale->get_domain_min();


    }

    void on_mouse_changed(int x, int y, int state) {
        ((Element *) plt_table)->mouse_changed(x, y, state);
    }

    void set_selected(bool s) {
        get_selected = [s]() { return s; };
    }

    void set_selected(std::function<bool()> f_selected) {
        get_selected = f_selected;
    }

    Scale *get_x_scale() {
        return x_scale;
    }

    Scale *get_y_scale() {
        return y_scale;
    }

    void set_x_axis_height(int s) {
        x_axis_height = s;
    }

    void set_y_axis_width(int s) {
        y_axis_width = s;
    }


    void add_content(Element *p) {
        content_group->add(p);
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;

        plt_table = new Table();

        Row *plt_row1 = new Row();
        plt_table->add(plt_row1);
        Row *plt_row2 = new Row();
        plt_table->add(plt_row2);

        y_axis->set_tick_height(y_axis_width);
        y_axis->left();
        plt_row1->add(
                new Cell(y_axis_width, y_scale->get_range_max() - y_scale->get_range_min(), (Element *) y_axis));

        plt_row1->add(new Cell(x_scale->get_range_max() - x_scale->get_range_min(),
                               y_scale->get_range_max() - y_scale->get_range_min(), (Element *) content_group));

        plt_row2->add(new Cell(y_axis_width, x_axis_height));

        x_axis->set_tick_height(x_axis_height);
        plt_row2->add(
                new Cell(x_scale->get_range_max() - x_scale->get_range_min(), x_axis_height, (Element *) x_axis));

        plt_table->build(x_offset, y_offset);

        width = ((Element *) plt_table)->get_width();
        height = ((Element *) plt_table)->get_height();
    }

    void enable_range_selection() {
        range_selection = true;
    }

    void disable_range_selection() {
        range_selection = false;
    }

    void paint() {
        plt_table->paint();

        if (get_selected()) {


            glBegin(GL_QUADS);
            glColor4f(context.select_r, context.select_g, context.select_b, 0.2);
            glVertex2i(x_offset, y_offset);
            glVertex2i(x_offset + width, y_offset);
            glVertex2i(x_offset + width, y_offset + height);
            glVertex2i(x_offset, y_offset + height);
            glEnd();

            glBegin(GL_LINES);
            glColor3f(context.select_r, context.select_g, context.select_b);

            glVertex2i(x_offset, y_offset);
            glVertex2i(x_offset + width, y_offset);

            glVertex2i(x_offset + width, y_offset);
            glVertex2i(x_offset + width, y_offset + height);

            glVertex2i(x_offset + width, y_offset + height);
            glVertex2i(x_offset, y_offset + height);

            glVertex2i(x_offset, y_offset + height);
            glVertex2i(x_offset, y_offset);
            glEnd();

        }

        if (range_selection && context.range_x_start != context.range_x_end &&
            context.range_y_start != context.range_y_end &&
            context.range_x == context.selected_i &&
            context.range_y == context.selected_j) {

            glBegin(GL_LINES);
            glColor3f(64. / 256., 128. / 256., 231. / 256.);

            glVertex2i(context.range_x_start, context.range_y_start);
            glVertex2i(context.range_x_end, context.range_y_start);

            glVertex2i(context.range_x_end, context.range_y_start);
            glVertex2i(context.range_x_end, context.range_y_end);

            glVertex2i(context.range_x_end, context.range_y_end);
            glVertex2i(context.range_x_start, context.range_y_end);

            glVertex2i(context.range_x_start, context.range_y_end);
            glVertex2i(context.range_x_start, context.range_y_start);
            glEnd();
        }
    }
};


class Button : Element {
private:
    float b_r = 0.;
    float b_g = 0.;
    float b_b = 0.;
    int arrow = 0;
    int hovered = false;

    char *text = nullptr;

    //layout
    int v_margin = 12;
    int h_margin = 16;
    int corner_cut = 1;
    int text_width = 0;
    int text_height = 12;
    void *font = GLUT_BITMAP_HELVETICA_12;

public:
    Button() {
        width = 36;
        height = 36;

        Button *b = this;

        this->add_move_listener([b]() {
            b->hovered = true;
        });
    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_background(float r, float g, float b) {
        b_r = r;
        b_g = g;
        b_b = b;
    }

    void top_arrow() {
        arrow = 1;
    }

    void right_arrow() {
        arrow = 2;
    }

    void bottom_arrow() {
        arrow = 3;
    }

    void left_arrow() {
        arrow = 4;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
    }

    void set_text(char *str) {
        text = str;
        text_width = glutBitmapLength(font, (const unsigned char *) text);
        width = text_width + 2 * v_margin;
    }

    void paint() {

        if (hovered) {
            if (x_offset <= context.mouse_x && context.mouse_x <= x_offset + width &&
                y_offset <= context.mouse_y && context.mouse_y <= y_offset + height) {
            } else {
                hovered = false;
            }
        }

        glBegin(GL_POLYGON);
        glColor4f(0., 0., 0., 0.2);

        glVertex2i(x_offset + corner_cut, y_offset - 1.);
        glVertex2i(x_offset, y_offset + corner_cut - 1.);

        glVertex2i(x_offset, y_offset + height - corner_cut - 1.);
        glVertex2i(x_offset + corner_cut, y_offset + height - 1.);

        glVertex2i(x_offset + width - corner_cut, y_offset + height - 1.);
        glVertex2i(x_offset + width, y_offset + height - corner_cut - 1.);

        glVertex2i(x_offset + width, y_offset + corner_cut - 1.);
        glVertex2i(x_offset + width - corner_cut, y_offset - 1.);

        glEnd();

        glBegin(GL_POLYGON);
        if (hovered) {
            glColor3f(b_r - 0.2, b_g - 0.2, b_b - 0.2);
        } else {
            glColor3f(b_r, b_g, b_b);
        }
        glVertex2i(x_offset + corner_cut, y_offset);
        glVertex2i(x_offset, y_offset + corner_cut);

        glVertex2i(x_offset, y_offset + height - corner_cut);
        glVertex2i(x_offset + corner_cut, y_offset + height);

        glVertex2i(x_offset + width - corner_cut, y_offset + height);
        glVertex2i(x_offset + width, y_offset + height - corner_cut);

        glVertex2i(x_offset + width, y_offset + corner_cut);
        glVertex2i(x_offset + width - corner_cut, y_offset);

        glEnd();


        // draw text
        if (text != nullptr) {

            int text_width = glutBitmapLength(font, (const unsigned char *) text);

            int x_extra = 0;
            x_extra = width - text_width;
            x_extra /= 2;

            int y_extra = 0;
            y_extra = height - text_height;
            y_extra /= 2;

            glColor3f(1., 1., 1.);

            glRasterPos2f(x_offset + x_extra, y_offset + y_extra);

            glutBitmapString(font, (const unsigned char *) text);
        }

        if (false) {
            // draw button
            int button_margin = 2;
            if (hovered) {
                if (x_offset <= context.mouse_x && context.mouse_x <= x_offset + width &&
                    y_offset <= context.mouse_y && context.mouse_y <= y_offset + height) {
                    glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides
                    glColor3f(b_r - .1, b_g - .1, b_b - .1); //Set the colour to red
                    glVertex2i(x_offset, y_offset);            //Draw the four corners of the rectangle
                    glVertex2i(x_offset, y_offset + height);
                    glVertex2i(x_offset + width, y_offset + height);
                    glVertex2i(x_offset + width, y_offset);
                    glEnd();
                } else {
                    hovered = false;
                }
            }

            if (!hovered) {
                glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides
                glColor3f(b_r, b_g, b_b);
                glVertex2i(x_offset, y_offset);
                glVertex2i(x_offset, y_offset + height);
                glVertex2i(x_offset + width, y_offset + height);
                glVertex2i(x_offset + width, y_offset);
                glEnd();

                /*
                glBegin(GL_QUADS);
                glLineWidth(button_margin);
                glColor3f(b_r - 0.2, b_g - 0.2, b_b - 0.2);
                glVertex2i(x_offset, y_offset + button_margin);
                glVertex2i(x_offset, y_offset);
                glVertex2i(x_offset + width, y_offset);
                glVertex2i(x_offset + width, y_offset + button_margin);
                glEnd();
                */
            }


            glBegin(GL_LINES);
            glLineWidth(1);
            glColor3f(b_r - 0.3, b_g - 0.3, b_b - 0.3);
            glVertex2i(x_offset, y_offset);
            glVertex2i(x_offset, y_offset + height);
            glVertex2i(x_offset, y_offset + height);
            glVertex2i(x_offset + width, y_offset + height);
            glVertex2i(x_offset + width, y_offset + height);
            glVertex2i(x_offset + width, y_offset);
            glVertex2i(x_offset + width, y_offset);
            glVertex2i(x_offset, y_offset);
            glEnd();


            glDisable(GL_POINT_SMOOTH);
            glPointSize(1);
            glBegin(GL_POINTS);
            glColor3f(1., 1., 1.);
            glVertex2i(x_offset, y_offset);
            glVertex2i(x_offset, y_offset + height);
            glVertex2i(x_offset + width, y_offset + height);
            glVertex2i(x_offset + width, y_offset);
            glEnd();



            // draw arrow
            int arrow_margin = 10;

            if (arrow == 1) {
                glBegin(GL_TRIANGLES);   //We want to draw a quad, i.e. shape with four sides
                glColor3f(1., 1., 1.); //Set the colour to
                glVertex2i(x_offset + width / 2, y_offset + height - arrow_margin);
                glVertex2i(x_offset + width / 2 - (height / 2 - arrow_margin), y_offset + arrow_margin);
                glVertex2i(x_offset + width / 2 + (height / 2 - arrow_margin), y_offset + arrow_margin);
                glEnd();
            }

            if (arrow == 2) {
                glBegin(GL_TRIANGLES);   //We want to draw a quad, i.e. shape with four sides
                glColor3f(1., 1., 1.); //Set the colour to
                glVertex2i(x_offset + width - arrow_margin, y_offset + height / 2);
                glVertex2i(x_offset + arrow_margin, y_offset + height / 2 - (width / 2 - arrow_margin));
                glVertex2i(x_offset + arrow_margin, y_offset + height / 2 + (width / 2 - arrow_margin));
                glEnd();
            }

            if (arrow == 3) {
                glBegin(GL_TRIANGLES);   //We want to draw a quad, i.e. shape with four sides
                glColor3f(1., 1., 1.); //Set the colour to
                glVertex2i(x_offset + width / 2, y_offset + arrow_margin);
                glVertex2i(x_offset + width / 2 - (height / 2 - arrow_margin), y_offset + height - arrow_margin);
                glVertex2i(x_offset + width / 2 + (height / 2 - arrow_margin), y_offset + height - arrow_margin);
                glEnd();
            }

            if (arrow == 4) {
                glBegin(GL_TRIANGLES);   //We want to draw a quad, i.e. shape with four sides
                glColor3f(1., 1., 1.); //Set the colour to
                glVertex2i(x_offset + arrow_margin, y_offset + height / 2);
                glVertex2i(x_offset + width - arrow_margin, y_offset + height / 2 - (width / 2 - arrow_margin));
                glVertex2i(x_offset + width - arrow_margin, y_offset + height / 2 + (width / 2 - arrow_margin));
                glEnd();
            }
        }
    }
};


void DrawCircle(float cx, float cy, float r, int num_segments, GLenum mode) {
    glBegin(mode);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle
        float x = r * cosf(theta);//calculate the x component
        float y = r * sinf(theta);//calculate the y component
        glVertex2f(x + cx, y + cy);//output vertex
    }
    glEnd();
}


class Action_Button : Element {
private:
    float b_r = 0.;
    float b_g = 0.;
    float b_b = 0.;
    int arrow = 0;
    int hovered = false;

    char *text = nullptr;

    //layout
    int margin = 16;
    int icon_width = 24;

public:
    Action_Button() {
        width = 56;
        height = 56;

        Action_Button *b = this;

        this->add_move_listener([b]() {
            b->hovered = true;
        });
    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_background(float r, float g, float b) {
        b_r = r;
        b_g = g;
        b_b = b;
    }

    void top_arrow() {
        arrow = 1;
    }

    void right_arrow() {
        arrow = 2;
    }

    void bottom_arrow() {
        arrow = 3;
    }

    void left_arrow() {
        arrow = 4;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
    }

    void set_text(char *str) {
        text = str;
    }

    void paint() {
        //DrawCircle(x_offset + width / 2., y_offset + height / 2., height / 2, 100, GL_POLYGON);


        if (hovered) {
            if (x_offset <= context.mouse_x && context.mouse_x <= x_offset + width &&
                y_offset <= context.mouse_y && context.mouse_y <= y_offset + height) {
            } else {
                hovered = false;
            }
        }

        float c_x = x_offset + width / 2.;
        float c_y = y_offset + height / 2.;

        glColor4f(0., 0., 0., 0.2);
        glPointSize(width + 1);
        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
        glVertex2f(c_x, c_y - 1);
        glEnd();

        glColor4f(0., 0., 0., 0.2);
        glPointSize(width + 1);
        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
        glVertex2f(c_x, c_y);
        glEnd();

        if (hovered) {
            glColor3f(b_r - 0.2, b_g - 0.2, b_b - 0.2);
        } else {
            glColor3f(b_r, b_g, b_b);
        }
        glPointSize(width);
        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
        glVertex2f(c_x, c_y);
        glEnd();

        // draw arrow
        glColor3f(1., 1., 1.); //Set the colour to
        if (arrow == 1) {

            glBegin(GL_POLYGON);
            //tip
            glVertex2f(c_x, c_y + icon_width / 2.);
            //right
            glVertex2f(c_x + icon_width / 2., c_y);
            glVertex2f(c_x + icon_width / 2. - 3, c_y - 3);
            //under
            glVertex2f(c_x, c_y + icon_width / 2. - 6);
            //left
            glVertex2f(c_x - icon_width / 2. + 3, c_y - 3);
            glVertex2f(c_x - icon_width / 2., c_y);
            glEnd();

            //line
            glBegin(GL_POLYGON);
            //bottom
            glVertex2f(c_x + 2, c_y - icon_width / 2. + 2);
            glVertex2f(c_x - 2, c_y - icon_width / 2. + 2);
            //top
            glVertex2f(c_x - 2, c_y + icon_width / 2. - 2);
            glVertex2f(c_x + 2, c_y + icon_width / 2. - 2);
            glEnd();
        } else if (arrow == 2) {
            glBegin(GL_POLYGON);
            //tip
            glVertex2f(c_x + icon_width / 2., c_y);
            //top
            glVertex2f(c_x, c_y + icon_width / 2.);
            glVertex2f(c_x - 3, c_y + icon_width / 2. - 3);
            //bottom
            glVertex2f(c_x + icon_width / 2. - 6, c_y);
            //left
            glVertex2f(c_x - 3, c_y - icon_width / 2. + 3);
            glVertex2f(c_x, c_y - icon_width / 2.);
            glEnd();

            //line
            glBegin(GL_POLYGON);
            //left
            glVertex2f(c_x - icon_width / 2. + 2, c_y + 2);
            glVertex2f(c_x - icon_width / 2. + 2, c_y - 2);
            //right
            glVertex2f(c_x + icon_width / 2. - 2, c_y - 2);
            glVertex2f(c_x + icon_width / 2. - 2, c_y + 2);
            glEnd();
        } else if (arrow == 3) {

            glBegin(GL_POLYGON);
            //tip
            glVertex2f(c_x, c_y - icon_width / 2.);
            //right
            glVertex2f(c_x + icon_width / 2., c_y);
            glVertex2f(c_x + icon_width / 2. - 3, c_y + 3);
            //under
            glVertex2f(c_x, c_y - icon_width / 2. + 6);
            //left
            glVertex2f(c_x - icon_width / 2. + 3, c_y + 3);
            glVertex2f(c_x - icon_width / 2., c_y);
            glEnd();

            //line
            glBegin(GL_POLYGON);
            //bottom
            glVertex2f(c_x + 2, c_y - icon_width / 2. + 2);
            glVertex2f(c_x - 2, c_y - icon_width / 2. + 2);
            //top
            glVertex2f(c_x - 2, c_y + icon_width / 2. - 2);
            glVertex2f(c_x + 2, c_y + icon_width / 2. - 2);
            glEnd();
        } else if (arrow == 4) {
            glBegin(GL_POLYGON);
            //tip
            glVertex2f(c_x - icon_width / 2., c_y);
            //top
            glVertex2f(c_x, c_y + icon_width / 2.);
            glVertex2f(c_x + 3, c_y + icon_width / 2. - 3);
            //bottom
            glVertex2f(c_x - icon_width / 2. + 6, c_y);
            //left
            glVertex2f(c_x + 3, c_y - icon_width / 2. + 3);
            glVertex2f(c_x, c_y - icon_width / 2.);
            glEnd();

            //line
            glBegin(GL_POLYGON);
            //left
            glVertex2f(c_x - icon_width / 2. + 2, c_y + 2);
            glVertex2f(c_x - icon_width / 2. + 2, c_y - 2);
            //right
            glVertex2f(c_x + icon_width / 2. - 2, c_y - 2);
            glVertex2f(c_x + icon_width / 2. - 2, c_y + 2);
            glEnd();
        }
    }
};

class Switch : Element {

private:
    float b_r = 0.;
    float b_g = 0.;
    float b_b = 0.;
    int arrow = 0;
    int hovered = false;

    std::function<bool()> get_on = nullptr;//[&]() {return context.use_GPU;};

    // layout
    int thumb_width = 36;
    int thumb_height = 16;
    int track_width = 20;
    int track_height = 20;

    int hover_counter = 0;

public:
    Switch(std::function<bool()> f_on) {
        width = 40;
        height = 20;
        get_on = f_on;

        Switch *s = this;

        this->add_move_listener([s]() {
            s->hovered = true;
        });
    }

    void on_mouse_changed(int x, int y, int state) {
    }

    void set_background(float r, float g, float b) {
        b_r = r;
        b_g = g;
        b_b = b;
    }

    void top_arrow() {
        arrow = 1;
    }

    void right_arrow() {
        arrow = 2;
    }

    void bottom_arrow() {
        arrow = 3;
    }

    void left_arrow() {
        arrow = 4;
    }

    void build(int new_x_offset, int new_y_offset) {
        x_offset = new_x_offset;
        y_offset = new_y_offset;
    }

    void paint() {

        if (hovered) {
            if (x_offset <= context.mouse_x && context.mouse_x <= x_offset + width &&
                y_offset <= context.mouse_y && context.mouse_y <= y_offset + height) {
            } else {
                hovered = false;
            }
        }

        if (hovered) {
            hover_counter += context.spf;
        } else {
            hover_counter = 0;
        }

        float c1_x = x_offset + track_width / 2.;
        float c1_y = y_offset + track_width / 2.;

        float c2_x = x_offset + width - track_width / 2.;
        float c2_y = y_offset + track_width / 2.;

        if (get_on()) {
            glColor3f(b_r, b_g, b_b);
        } else {
            glColor3f(155 / 255., 155 / 255., 155 / 255.);
        }

        glBegin(GL_QUADS);   //We want to draw a quad, i.e. shape with four sides
        glVertex2f(c1_x, c1_y - thumb_height / 2.);
        glVertex2f(c1_x, c1_y + thumb_height / 2.);
        glVertex2f(c2_x, c2_y + thumb_height / 2.);
        glVertex2f(c2_x, c2_y - thumb_height / 2.);
        glEnd();

        DrawCircle(c1_x, c1_y, thumb_height / 2., 100, GL_POLYGON);
        DrawCircle(c2_x, c2_y, thumb_height / 2., 100, GL_POLYGON);

        float hover_radius =
                track_width < track_width / 2. + track_width / (500. / hover_counter) ? track_width :
                track_width / 2. +
                track_width /
                (500. /
                 hover_counter +
                 .5);


        if (get_on()) {
            glColor4f(0., 0., 0., 0.2);
            DrawCircle(c2_x, c2_y - 1, track_width / 2. + 1., 100, GL_POLYGON);
            glColor4f(0., 0., 0., 0.2);
            DrawCircle(c2_x, c2_y, track_width / 2. + 1., 100, GL_POLYGON);
            glColor3f(b_r - 0.2, b_g - 0.2, b_b - 0.2);
            DrawCircle(c2_x, c2_y, track_width / 2., 100, GL_POLYGON);
            if (hovered) {
                glColor4f(b_r - 0.1, b_g - 0.1, b_b - 0.1, 0.2);
                DrawCircle(c2_x, c2_y, hover_radius, 100, GL_POLYGON);
            }
        } else {
            glColor4f(0., 0., 0., 0.2);
            DrawCircle(c1_x, c1_y - 1, track_width / 2. + 1., 100, GL_POLYGON);
            glColor4f(0., 0., 0., 0.2);
            DrawCircle(c1_x, c1_y, track_width / 2. + 1., 100, GL_POLYGON);
            glColor3f(1., 1., 1.);
            DrawCircle(c1_x, c1_y, track_width / 2., 100, GL_POLYGON);
            if (hovered) {
                glColor4f(0., 0., 0., 0.2);
                DrawCircle(c1_x, c1_y, hover_radius, 100, GL_POLYGON);
            }

        }
    }
};

/*void createPoints(GLuint* points, struct cudaGraphicsResource** cuda_points,
	unsigned int cuda_points_flags)
{
	if (points) {
		// create buffer object
		glGenBuffers(1, points);
		glBindBuffer(GL_ARRAY_BUFFER, *points);

		// initialize buffer object
		unsigned int size = n * 2 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// register this buffer object with CUDA
		cudaGraphicsGLRegisterBuffer(cuda_points, *points, cuda_points_flags);
	}
}

void runCuda()
{
	cudaGraphicsMapResources(1, &cuBuffer, 0);
	size_t num_bytes;
	float* cuVertex;
	cudaGraphicsResourceGetMappedPointer
	((void**)&cuVertex, &num_bytes, cuBuffer);

	kernel<<<1,n>>>(cuVertex, n);


	// unmap buffer object
	cudaGraphicsUnmapResources(1, &cuBuffer);
}*/

void set_selected_dims(int i, int j) {
    context.selected_i = i;
    context.selected_j = j;
    context.changed();
}

bool match_selected(int i, int j) {
    return context.selected_i == i && context.selected_j == j;
}

Table *build_scatter_plot_matrix(int scatter_plot_matrix_size) {
    //todo it looks like you have swapped around on the axis????
    int plot_full_size = (scatter_plot_matrix_size) / context.d - 5;
    int plot_full_margin = 2;
    bool heatmap = true;
    bool density = false;

    Table *scatter_plot_matrix = new Table();
    scatter_plot_matrix->set_spacing(5);

    for (int j = 0; j < context.d; j++) {
        Row *scatter_plot_row = new Row();
        scatter_plot_matrix->add(scatter_plot_row);
        for (int i = 0; i < context.d; i++) {
            Cell *scatter_plot_cell = new Cell(plot_full_size, plot_full_size);
            scatter_plot_row->add(scatter_plot_cell);
            if (i != j) {
                Scale *x_scale = new Scale(context.min_values[i], context.max_values[i], 0,
                                           plot_full_size - plot_full_margin);
                Scale *y_scale = new Scale(context.min_values[j], context.max_values[j], 0,
                                           plot_full_size - plot_full_margin);
                Plot *plt = new Plot(x_scale, y_scale);
                plt->set_selected([i, j]() { return match_selected(i, j) || match_selected(j, i); });
                ((Element *) plt)->add_click_listener([i, j]() {
                    set_selected_dims(i, j);
                });

                if (i > j) {//(i > j || (!heatmap && !density)) {
                    Points *pts = new Points(x_scale, y_scale, context.d_points_in, context.h_points_in, context.n,
                                             context.d);
                    pts->set_dims(i, j);
                    //pts->set_sampling_interval(5);
                    plt->add_content(pts);

                    scatter_plot_cell->set_content((Element *) plt);
                } else if (i > j) {

                    Small_points *pts = new Small_points(x_scale, y_scale, context.d_points_in, context.h_points_in,
                                                         context.n, context.d);
                    pts->set_dims(i, j);
                    plt->add_content((Element *) pts);

                    scatter_plot_cell->set_content((Element *) plt);
                } else if (true) {//(i < j) {

                    if (heatmap) {
                        Heatmap *pts = new Heatmap(x_scale, y_scale, context.d_points_in, context.h_points_in,
                                                   context.n, context.d);
                        pts->set_dims(i, j);
                        plt->add_content((Element *) pts);

                        scatter_plot_cell->set_content((Element *) plt);
                    } else if (density) {
                        Density *pts = new Density(x_scale, y_scale, context.d_points_in, context.n, context.d);
                        pts->set_dims(i, j);
                        //plt->add_content((Content*) pts);
                        plt->add_content((Element *) pts);

                        scatter_plot_cell->set_content((Element *) plt);
                    }
                }
            } else {
                Text *text_l = new Text(plot_full_size, plot_full_size);
                text_l->set_text([i]() {
                    std::string str = "D" + std::to_string(i);
                    char *cstr = new char[str.length() + 1];
                    strcpy(cstr, str.c_str());

                    return cstr;
                });
                text_l->set_font(GLUT_BITMAP_HELVETICA_18);
                text_l->h_align_center();
                text_l->v_align_center();
                text_l->set_color([i](Text *t) {
                    if (i == context.selected_i || i == context.selected_j) {
                        t->r = context.select_r;
                        t->g = context.select_g;
                        t->b = context.select_b;
                    } else {
                        t->r = 0.;
                        t->g = 0.;
                        t->b = 0.;
                    }
                });


                scatter_plot_cell->set_content((Element *) text_l);
            }
        }
    }

    return scatter_plot_matrix;
}

void update_k_center(int change) {
    context.k_center += change;
    if (context.k_center < 3) {
        context.k_center = 3;
    }
    if (context.k_center >= context.k_max) {
        context.k_center = context.k_max - 1;
    }
    context.changed();
}

void update_l_center(int change) {
    context.l_center += change;
    if (context.l_center < 3) {
        context.l_center = 3;
    }
    if (context.l_center > context.d - 1) {
        context.l_center = context.d - 1;
    }
    context.changed();
}

void update_k(int extra) {
    context.k = context.k_center + extra;

//    context.SC = SilhouetteCoefficient(context.h_points_in, context.get_result(context.k, context.l).get_h_C(), context.d, context.n, context.k);
//    printf("SC: %f\n", context.SC);
//    context.CH = CalinskiHarabaszScore(context.h_points_in, context.get_result(context.k, context.l).get_h_C(),
//                                       context.d, context.n, context.k);
//    printf("CH: %f\n", context.CH);

    context.changed();
}

void update_l(int extra) {
    context.l = context.l_center + extra;

//    context.SC = SilhouetteCoefficient(context.h_points_in, context.get_result(context.k, context.l).get_h_C(), context.d, context.n, context.k);
//    printf("SC: %f\n", context.SC);
//    context.CH = CalinskiHarabaszScore(context.h_points_in, context.get_result(context.k, context.l).get_h_C(),
//                                       context.d, context.n, context.k);
//    printf("CH: %f\n", context.CH);

    context.changed();
}

int get_l_center(int extra) {
    return context.l_center + extra;
}

int get_k_center(int extra) {
    return context.k_center + extra;
}

int get_l() {
    return context.l;
}

int get_k() {
    return context.k;
}

Table *build_scatter_right_menu(int width, int height) {
    int label_width = 30;
    width -= label_width;

    Table *menu = new Table();
    menu->set_spacing(5);


    int button_height = (height - width * 3 - 5 * 4) / 2;
    Action_Button *top_button = new Action_Button();
    top_button->set_background(context.button_r, context.button_g, context.button_b);
    top_button->top_arrow();
    ((Element *) top_button)->add_click_listener([]() {
        update_k_center(1);
    });
    Row *row = new Row();
    row->add(new Cell(label_width, button_height));
    Cell *top_cell = new Cell(width, button_height, (Element *) top_button);
    top_cell->h_align_center();
    top_cell->v_align_center();
    row->add(top_cell);
    menu->add(row);


    for (int i = 0; i < 3; i++) {
        Scale *x_scale = new Scale(context.min_values[context.selected_i], context.max_values[context.selected_i],
                                   0,
                                   width);
        Scale *y_scale = new Scale(context.min_values[context.selected_j], context.max_values[context.selected_j],
                                   0,
                                   width);
        Plot *plt = new Plot(x_scale, y_scale);
        plt->set_selected([i]() { return get_k_center(1 - i) == get_k(); });
        ((Element *) plt)->add_click_listener([i]() { update_k(1 - i); });
        Row *row = new Row();

        Points *pts = new Points(x_scale, y_scale, context.d_points_in, context.h_points_in, context.n, context.d);
        pts->set_dims(context.selected_i, context.selected_j);
        pts->set_k([i]() { return get_k_center(1 - i); });
        //plt->add_content((Content*)pts);
        plt->add_content(pts);

        context.add_change_listener([plt, pts]() {
            pts->set_dims(context.selected_i, context.selected_j);
            plt->get_x_scale()->domain(context.min_values[context.selected_i],
                                       context.max_values[context.selected_i]);
            plt->get_y_scale()->domain(context.min_values[context.selected_j],
                                       context.max_values[context.selected_j]);
        });

        Text *text_k = new Text(label_width, width);
        text_k->set_text([i]() {
            std::string str = "k=" + std::to_string(get_k_center(1 - i));
            char *cstr = new char[str.length() + 1];
            strcpy(cstr, str.c_str());

            return cstr;
        });
        text_k->set_font(GLUT_BITMAP_HELVETICA_18);
        text_k->h_align_center();
        text_k->v_align_center();
        row->add(new Cell(label_width, width, (Element *) text_k));

        row->add(new Cell(width, width, (Element *) plt));
        menu->add(row);

    }

    Action_Button *bottom_button = new Action_Button();
    bottom_button->set_background(context.button_r, context.button_g, context.button_b);
    bottom_button->bottom_arrow();
    ((Element *) bottom_button)->add_click_listener([]() {
        update_k_center(-1);
    });
    row = new Row();
    row->add(new Cell(label_width, button_height));
    Cell *bottom_cell = new Cell(width, button_height, (Element *) bottom_button);
    bottom_cell->h_align_center();
    bottom_cell->v_align_center();
    row->add(bottom_cell);
    menu->add(row);
    return menu;
}

Table *build_scatter_bottom_menu(int width, int height) {

    int label_height = 30;

    height -= label_height;

    Table *menu = new Table();
    menu->set_spacing(5);
    Row *menu_labels = new Row();
    menu->add(menu_labels);
    Row *menu_row = new Row();
    menu->add(menu_row);


    int button_width = (width - height * 3 - 5 * 4) / 2;
    Action_Button *left_button = new Action_Button();
    left_button->set_background(context.button_r, context.button_g, context.button_b);
    left_button->left_arrow();
    ((Element *) left_button)->add_click_listener([]() {
        update_l_center(-1);
    });
    Cell *left_cell = new Cell(button_width, height, (Element *) left_button);
    left_cell->h_align_center();
    left_cell->v_align_center();
    menu_row->add(left_cell);

    menu_labels->add(new Cell(button_width, label_height));

    for (int i = 0; i < 3; i++) {
        Scale *x_scale = new Scale(context.min_values[context.selected_i], context.max_values[context.selected_i],
                                   0,
                                   height);
        Scale *y_scale = new Scale(context.min_values[context.selected_j], context.max_values[context.selected_j],
                                   0,
                                   height);
        Plot *plt = new Plot(x_scale, y_scale);
        plt->set_selected([i]() { return get_l_center(-1 + i) == get_l(); });
        ((Element *) plt)->add_click_listener([i]() { update_l(-1 + i); });
        menu_row->add(new Cell(height, height, (Element *) plt));

        Points *pts = new Points(x_scale, y_scale, context.d_points_in, context.h_points_in, context.n, context.d);
        pts->set_dims(context.selected_i, context.selected_j);
        pts->set_l([i]() { return get_l_center(-1 + i); });
        //plt->add_content((Content*)pts);
        plt->add_content(pts);


        context.add_change_listener([pts, x_scale, y_scale]() {
            pts->set_dims(context.selected_i, context.selected_j);
            x_scale->domain(context.min_values[context.selected_i], context.max_values[context.selected_i]);
            y_scale->domain(context.min_values[context.selected_j], context.max_values[context.selected_j]);
        });


        Text *text_l = new Text(height, label_height);
        text_l->set_text([i]() {
            std::string str = "l=" + std::to_string(get_l_center(-1 + i));
            char *cstr = new char[str.length() + 1];
            strcpy(cstr, str.c_str());

            return cstr;
        });
        text_l->set_font(GLUT_BITMAP_HELVETICA_18);
        text_l->h_align_center();
        text_l->v_align_center();
        menu_labels->add(new Cell(height, label_height, (Element *) text_l));

    }

    Action_Button *right_button = new Action_Button();
    right_button->set_background(context.button_r, context.button_g, context.button_b);
    right_button->right_arrow();
    ((Element *) right_button)->add_click_listener([]() {
        update_l_center(1);
    });

    Cell *right_cell = new Cell(button_width, height, (Element *) right_button);
    right_cell->h_align_center();
    right_cell->v_align_center();
    menu_row->add(right_cell);
    menu_labels->add(new Cell(button_width, label_height));

    return menu;
}

Table *build_cluster_info(int row_2_height) {
    int axis_width = 20;

    Table *info_table = new Table();

    Text *text_kl = new Text(row_2_height, row_2_height * 1. / 5.);
    text_kl->set_text([&]() {
        std::string str = "k=" + std::to_string(context.k) + ", l=" + std::to_string(context.l);
        char *cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    });
    text_kl->set_font(GLUT_BITMAP_HELVETICA_18);
    text_kl->h_align_center();
    text_kl->v_align_center();
    //cell22->set_content((Element*)text_kl);
    info_table->add(new Row(new Cell(row_2_height, row_2_height * 1. / 5., (Element *) text_kl)));


//    Text *text_sc = new Text(row_2_height, row_2_height * 1. / 5.);
//    text_sc->set_text([&]() {
//        std::string str = "SC=" + std::to_string(context.SC);
//        char *cstr = new char[str.length() + 1];
//        strcpy(cstr, str.c_str());
//        return cstr;
//    });
//    text_sc->set_font(GLUT_BITMAP_HELVETICA_18);
//    info_table->add(new Row(new Cell(row_2_height, row_2_height * 1. / 5., (Element *) text_sc)));


//    Text *text_ch = new Text(row_2_height, row_2_height * 1. / 5.);
//    text_ch->set_text([&]() {
//        std::string str = "CH=" + std::to_string(context.CH);
//        char *cstr = new char[str.length() + 1];
//        strcpy(cstr, str.c_str());
//        return cstr;
//    });
//    text_ch->set_font(GLUT_BITMAP_HELVETICA_18);
//    info_table->add(new Row(new Cell(row_2_height, row_2_height * 1. / 5., (Element *) text_ch)));


    Scale *bar_x_scale = new Scale(0, context.k + 1, 0, row_2_height);
    Scale *bar_y_scale = new Scale(0, context.n, 0, row_2_height * 4. / 5.);
    Plot *bar_chart = new Plot(bar_x_scale, bar_y_scale);
//    bar_chart->get_y_axis()->enable_tick_label();
//    bar_chart->set_y_axis_width(axis_width);
//    bar_chart->get_x_axis()->enable_tick_label();
//    bar_chart->set_x_axis_height(axis_width);
    Bars *bars = new Bars(bar_x_scale, bar_y_scale);
    bars->set_count(context.k + 1);
    context.add_change_listener([bars]() { bars->set_count(context.k + 1); });
    bars->set_spacing(5);
    bars->set_data([&]() -> int * { return context.get_result(context.k, context.l).d_C; }, context.n, 1);
    bars->set_transform([]__device__(int *d_C, int d, int p, int x_dim, int y_dim) -> int { return d_C[p] + 1; });
    bars->add_bar_click_listener([&](int id) {
        if (context.selected_c == id - 1) {
            context.selected_c = -2;
        } else {
            context.selected_c = id - 1;
        }
    });
    bar_chart->add_content(bars);


    info_table->add(new Row(new Cell(row_2_height, row_2_height * 4. / 5., (Element *) bar_chart)));

    return info_table;
}

Table *build_left(int body_height) {


    Table *left_side = new Table();
    left_side->set_spacing(10);

    int row_1_height = 4 * (body_height - 10) / 5;
    int row_2_height = (body_height - 10) / 5;

    Row *row1 = new Row();
    left_side->add(row1);
    Cell *cell11 = new Cell(row_1_height, row_1_height);
    row1->add(cell11);
    Cell *cell12 = new Cell(row_2_height, row_1_height);
    row1->add(cell12);

    Row *row2 = new Row();
    left_side->add(row2);
    Cell *cell21 = new Cell(row_1_height, row_2_height);
    row2->add(cell21);
    Cell *cell22 = new Cell(row_2_height, row_2_height);
    row2->add(cell22);


    cell11->set_content((Element *) build_scatter_plot_matrix(row_1_height));
    cell12->set_content((Element *) build_scatter_right_menu(row_2_height, row_1_height));
    cell21->set_content((Element *) build_scatter_bottom_menu(row_1_height, row_2_height));
    cell22->set_content((Element *) build_cluster_info(row_2_height));

    return left_side;
}

Table *build_right(int body_height, int body_margin, int right_side_width) {

    //common sizes
    int right_side_spacing = 5;

    //menu sizes
    int menu_height = 36;

    //plot sizes
    int scatter_plot_width = min(right_side_width,
                                 body_height - 2 * body_margin - menu_height - right_side_spacing);
    int scatter_plot_height = scatter_plot_width;
    int axis_width = 30;
    int point_size = 5;


    Table *right_side = new Table();
    right_side->set_spacing(right_side_spacing);


    /// Big scatter plot
    //right_side->add(new Row(new Cell(right_side_width, (body_height- right_side_width)/2)));

    Scale *x_scale = new Scale(context.min_values[context.selected_i], context.max_values[context.selected_i], 0,
                               scatter_plot_width - axis_width);
    Scale *y_scale = new Scale(context.min_values[context.selected_j], context.max_values[context.selected_j], 0,
                               scatter_plot_height - axis_width);
    Plot *plt = new Plot(x_scale, y_scale);
    plt->get_x_axis()->enable_tick_label();
    plt->get_y_axis()->enable_tick_label();

    Points *pts = new Points(x_scale, y_scale, context.d_points_in, context.h_points_in, context.n, context.d);
    pts->set_size(point_size);
    pts->set_dims(context.selected_i, context.selected_j);
    //plt->add_content((Content*)pts);
    plt->add_content(pts);
    plt->enable_range_selection();
    plt->set_x_axis_height(axis_width);
    plt->set_y_axis_width(axis_width);

    context.add_change_listener([plt, pts]() {
        pts->set_dims(context.selected_i, context.selected_j);
        plt->get_x_scale()->domain(context.min_values[context.selected_i], context.max_values[context.selected_i]);
        plt->get_y_scale()->domain(context.min_values[context.selected_j], context.max_values[context.selected_j]);
    });
    right_side->add(new Row(new Cell(scatter_plot_width, scatter_plot_height, (Element *) plt)));

    //right_side->add(new Row(new Cell(right_side_width, (body_height - right_side_width) / 2)));


    /// menu
    Table *menu = new Table();
    menu->set_spacing(right_side_spacing);

    right_side->add(new Row(new Cell(right_side_width, menu_height, (Element *) menu)));

    Row *menu_row = new Row();
    menu->add(menu_row);


    int cell_width = right_side_width / 4;

    Button *re_button = new Button();
    re_button->set_background(context.button_r, context.button_g, context.button_b);
    re_button->set_text("Re-compute result!");
    ((Element *) re_button)->add_click_listener([]() {
        context.model->clear_results();
    });


    Cell *menu_row_cell1 = new Cell(cell_width, menu_height, (Element *) re_button);
    menu_row_cell1->v_align_center();
    menu_row_cell1->h_align_center();
    menu_row->add(menu_row_cell1);

    Text *GPU_vis_text = new Text(cell_width - right_side_spacing, menu_height);
    GPU_vis_text->set_text([]() {
        std::string str = "Use GPU-visualization:";
        char *cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    });
    GPU_vis_text->h_align_right();
    GPU_vis_text->v_align_center();
    GPU_vis_text->set_font(GLUT_BITMAP_HELVETICA_18);
    GPU_vis_text->auto_size();

    Cell *menu_row_cell2 = new Cell(cell_width - right_side_spacing, menu_height, (Element *) GPU_vis_text);
    menu_row_cell2->h_align_right();
    menu_row_cell2->v_align_center();
    menu_row->add(menu_row_cell2);

    Switch *s = new Switch([&]() { return context.use_GPU; });
    s->set_background(context.button_r, context.button_g, context.button_b);
    ((Element *) s)->add_click_listener([&]() {
        context.use_GPU = !context.use_GPU;
    });

    Cell *menu_row_cell3 = new Cell(cell_width / 2 - 2 * right_side_spacing, menu_height, (Element *) s);
    menu_row->add(menu_row_cell3);
    menu_row_cell3->v_align_center();

    Text *SPF_text = new Text(cell_width - right_side_spacing, menu_height);
    SPF_text->set_text([]() {
        std::string str = std::to_string(context.spf) + " ms per frame.";
        char *cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    });
    SPF_text->h_align_right();
    SPF_text->v_align_center();
    SPF_text->set_font(GLUT_BITMAP_HELVETICA_18);
    SPF_text->auto_size();

    Cell *menu_row_cell4 = new Cell(cell_width - right_side_spacing, menu_height, (Element *) SPF_text);
    menu_row_cell4->h_align_right();
    menu_row_cell4->v_align_center();
    menu_row->add(menu_row_cell4);

    return right_side;
}

void build_display() {

    int body_margin = 10;
    int body_height = context.window_height - 2 * body_margin;
    int right_side_width = context.window_width - context.window_height - 2 * body_margin;


    context.body = new Table();
    context.body->set_spacing(10);
    Row *body_row = new Row();
    context.body->add(body_row);

    body_row->add(new Cell(body_height, body_height, (Element *) build_left(body_height)));
    body_row->add(new Cell(right_side_width, body_height,
                           (Element *) build_right(body_height, body_margin, right_side_width)));


    context.body->build(body_margin, body_margin);

}


void run_proclus() {
    int a = 40;
    int b = 10;
    float min_deviation = 0.5;
    int termination_rounds = 20;

    context.result = GPU_PROCLUS_SAVE(context.d_points_in, context.n, context.d, context.k, context.l, a, b,
                                      min_deviation, termination_rounds, false);
}

void display() {  // Display function will draw the image.
    context.window_width = glutGet(GLUT_WINDOW_WIDTH);
    context.window_height = glutGet(GLUT_WINDOW_HEIGHT);

    glClearColor(1, 1, 1, 1);  // (In fact, this is the default.)
    glClear(GL_COLOR_BUFFER_BIT);
    context.body->paint();

    if (context.tooltip != nullptr) {
        context.tooltip();
        context.tooltip = nullptr;
    }

    glutSwapBuffers(); // Required to copy color buffer onto the screen.
    context.t += 0.01;
    context.t = context.t > 1. ? 0. : context.t;

}


void FPS(void) {


    //static GLint Frames = 0;         // frames averaged over 1000mS
    static GLuint Clock;             // [milliSeconds]
    static GLuint PreviousClock = 0; // [milliSeconds]
    //static GLuint NextClock = 0;     // [milliSeconds]

    //++Frames;
    Clock = glutGet(GLUT_ELAPSED_TIME); //has limited resolution, so average over 1000mS
    //if (Clock < NextClock) return;

    context.spf = Clock - PreviousClock;
    //printf("Seconds per frame: %f\n", ((float)Clock - PreviousClock) / ((float)Frames));

    PreviousClock = Clock;
    //NextClock = Clock + 100; // 1000mS=1S in the future
    //Frames = 0;
}

void timerEvent(int value) {
    if (glutGetWindow()) {

        FPS(); //only call once per frame loop to measure FPS
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
        glutPostRedisplay();

    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////

void mouse_move(int x, int y) {

    y = context.window_height - y;

    context.mouse_x = x;
    context.mouse_y = y;

    ((Element *) context.body)->mouse_changed(x, y, MOUSE_MOVE);

}

void mouse(int button, int state, int x, int y) {
    y = context.window_height - y;


    context.mouse_x = x;
    context.mouse_y = y;


    if (state == GLUT_DOWN) {
        context.mouse_buttons |= 1 << button;
        ((Element *) context.body)->mouse_changed(x, y, MOUSE_DOWN);
    } else if (state == GLUT_UP) {
        context.mouse_buttons = 0;
        ((Element *) context.body)->mouse_changed(x, y, MOUSE_UP);
    }

    context.mouse_old_x = x;
    context.mouse_old_y = y;

}

/*void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}*/



int main(int argc, char **argv) {  // Initialize GLUT

    char *file = "data/X10_000.csv";
    if (argc > 1) {
        file = argv[1];
    }

    std::ifstream in(file);
    std::vector <std::vector<float>> fields;

    if (in) {
        std::string line;

        while (std::getline(in, line)) {
            std::stringstream sep(line);
            std::string field;

            fields.push_back(std::vector<float>());

            while (getline(sep, field, ',')) {
                fields.back().push_back(stof(field));
            }
        }
    } else {
        printf("data not found!");
        return 0;
    }

    if (fields.size() > 0 && fields[0].size() > 0) {
        context.n = fields.size();
        context.d = fields[0].size();

        printf("n: %d, d: %d\n", context.n, context.d);

        context.min_values = new float[context.d];
        context.max_values = new float[context.d];
        for (int j = 0; j < context.d; j++) {
            context.min_values[j] = fields[0][j];
            context.max_values[j] = fields[0][j];
        }
        context.h_points_in = new float[context.n * context.d];
        for (int i = 0; i < context.n; i++) {
            for (int j = 0; j < context.d; j++) {
                context.h_points_in[i * context.d + j] = fields[i][j];
                if (fields[i][j] > context.max_values[j]) {
                    context.max_values[j] = fields[i][j];
                }
                if (fields[i][j] < context.min_values[j]) {
                    context.min_values[j] = fields[i][j];
                }
            }
        }
        cudaMalloc(&(context.d_points_in), context.n * context.d * sizeof(float));
        cudaMemcpy(context.d_points_in, context.h_points_in, context.n * context.d * sizeof(float),
                   cudaMemcpyHostToDevice);

    }

    //run_proclus();

    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_SINGLE);    // Use single color buffer and no depth buffer.
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA);
    glutInitWindowSize(context.window_width,
                       context.window_height);         // Size of display area, in pixels.  + 17
    glutInitWindowPosition(0, 0);     // Location of window in screen coordinates.
    glutCreateWindow("AVID"); // Parameter is window title.

    //enable alpha values
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glewInit();
    //glutFullScreen();
    build_display();
//    context.CH = CalinskiHarabaszScore(context.h_points_in, context.get_result(context.k, context.l).get_h_C(),
//                                       context.d, context.n, context.k);
    glutDisplayFunc(display);            // Called when the window needs to be redrawn.
    glutMouseFunc(mouse);
    glutMotionFunc(mouse_move);
    glutPassiveMotionFunc(mouse_move);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, context.window_width, 0.0, context.window_height, 0.0, 1.0);


    // create
    //Gluint glBuffer;
    glGenBuffers(1, &(context.glBuffer));
    glBindBuffer(GL_ARRAY_BUFFER, context.glBuffer);
    glBufferData(GL_ARRAY_BUFFER, context.n * (2 + 4) * sizeof(float), NULL,
                 GL_DYNAMIC_DRAW); //making room for location and color

    cudaGLSetGLDevice(0); // explicitly set device 0
    //struct cudaGraphicsResource* cuBuffer;
    cudaGraphicsGLRegisterBuffer
            (&(context.cuBuffer), context.glBuffer, cudaGraphicsMapFlagsWriteDiscard);
    // cudaGraphicsMapFlagsWriteDiscard:
    // CUDA will only write and will not read from this resource



    glutMainLoop(); // Run the event loop!  This function does not return.
    // Program ends when user closes the window.


    return 0;

}
