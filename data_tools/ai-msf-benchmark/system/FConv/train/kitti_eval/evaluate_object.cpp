#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <strings.h>
#include <assert.h>

#include <dirent.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

#include "mail.h"

BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)

typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > Polygon;


using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// holds the number of test images on the server
#ifndef OFFLINE_EVAL
const int32_t N_TESTIMAGES = 7518;
#endif

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2};

// evaluation metrics: image, ground or 3D
enum METRIC{IMAGE=0, GROUND=1, BOX3D=2};

// evaluation parameter
const int32_t MIN_HEIGHT[3]     = {40, 25, 25};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3]  = {0, 1, 2};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[3] = {0.15, 0.3, 0.5}; // maximum truncation level of the groundtruth used for evaluation

// evaluated object classes
enum CLASSES{CAR=0, PEDESTRIAN=1, CYCLIST=2};
const int NUM_CLASS = 3;

// parameters varying per class
vector<string> CLASS_NAMES;
vector<string> CLASS_NAMES_CAP;
// the minimum overlap required for 2D evaluation on the image/ground plane and 3D evaluation
const double MIN_OVERLAP[3][3] = {{0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}};

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("pedestrian");
  CLASS_NAMES.push_back("cyclist");
  CLASS_NAMES_CAP.push_back("Car");
  CLASS_NAMES_CAP.push_back("Pedestrian");
  CLASS_NAMES_CAP.push_back("Cyclist");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  double  ry;
  double  t1, t2, t3;
  double  h, w, l;
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};


/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/
void update_eval_flags(const tDetection &d, vector<bool> &eval_image,
        vector<bool> &eval_ground, vector<bool> &eval_3d) {
  for (int c = 0; c < NUM_CLASS; ++c) {
    bool matching_class = !strcasecmp(d.box.type.c_str(), CLASS_NAMES[c].c_str()) ||
                          !strcasecmp(d.box.type.c_str(), CLASS_NAMES_CAP[c].c_str());
    if (!matching_class) continue;
    if (d.box.x1 >= 0) eval_image[c] = true;
    if (d.t1 != -1000 && d.t3 != -1000 && d.w > 0 && d.l > 0) eval_ground[c] = true;
    if (d.t1 != -1000 && d.t2 != -1000 && d.t3 != -1000 &&
        d.h > 0 && d.w > 0 && d.l > 0) eval_3d[c] = true;
    break;
  }
}

vector<tDetection> loadDetections(string file_name, bool &compute_aos,
        vector<bool> &eval_image, vector<bool> &eval_ground,
        vector<bool> &eval_3d, bool &success) {
  vector<tDetection> detections;
  std::ifstream input(file_name);
  if (!input) {
    success = false;
    return detections;
  }
  tDetection detection;
  double unused_value;
  string type;
  while (input >> type >> unused_value >> unused_value >> detection.box.alpha
         >> detection.box.x1 >> detection.box.y1 >> detection.box.x2 >> detection.box.y2
         >> detection.h >> detection.w >> detection.l >> detection.t1 >> detection.t2
         >> detection.t3 >> detection.ry >> detection.thresh) {
    detection.box.type = type;
    detections.push_back(detection);
    if (detection.box.alpha == -10) compute_aos = false;
    update_eval_flags(detection, eval_image, eval_ground, eval_3d);
  }
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  std::ifstream input(file_name);
  if (!input) {
    success = false;
    return groundtruth;
  }
  while (input) {
    tGroundtruth g;
    string str;
    if (input >> str >> g.truncation >> g.occlusion >> g.box.alpha
              >> g.box.x1 >> g.box.y1 >> g.box.x2 >> g.box.y2
              >> g.h >> g.w >> g.l >> g.t1 >> g.t2 >> g.t3 >> g.ry) {
      g.box.type = str;
      groundtruth.push_back(g);
    }
  }
  success = true;
  return groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double imageBoxOverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

inline double imageBoxOverlap(tDetection a, tGroundtruth b, int32_t criterion=-1){
  return imageBoxOverlap(a.box, b.box, criterion);
}

// compute polygon of an oriented bounding box
template <typename T>
Polygon toPolygon(const T& g) {
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(g.ry); mref(0, 1) = sin(g.ry);
    mref(1, 0) = -sin(g.ry); mref(1, 1) = cos(g.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {g.l / 2, g.l / 2, -g.l / 2, -g.l / 2,
                     g.w / 2, -g.w / 2, -g.w / 2, g.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += g.t1;
        gc(1, i) += g.t3;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}

// measure overlap between bird's eye view bounding boxes, parametrized by (ry, l, w, tx, tz)
inline double groundBoxOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double inter_area = in.empty() ? 0 : area(in.front());
    double union_area = area(un.front());
    if (union_area <= 0.0)
        return 0.0;
    double o;
    if(criterion==-1)     // union
        o = inter_area / std::max(union_area, 1e-12);
    else if(criterion==0) // bbox_a
        o = inter_area / area(dp);
    else if(criterion==1) // bbox_b
        o = inter_area / area(gp);

    return o;
}

// measure overlap between 3D bounding boxes, parametrized by (ry, h, w, l, tx, ty, tz)
inline double box3DOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double ymax = min(d.t2, g.t2);
    double ymin = max(d.t2 - d.h, g.t2 - g.h);

    double inter_area = in.empty() ? 0 : area(in.front());
    double inter_vol = inter_area * max(0.0, ymax - ymin);

    double det_vol = d.h * d.l * d.w;
    double gt_vol = g.h * g.l * g.w;

    double denominator = det_vol + gt_vol - inter_vol;
    if (criterion == 0) denominator = det_vol;
    else if (criterion == 1) denominator = gt_vol;
    return denominator > 0.0 ? inter_vol / denominator : 0.0;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;
  if (n_groundtruth <= 0.0)
    return t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1) / std::max(n_groundtruth, 1.0);
    if(i<(v.size()-1))
      r_recall = (double)(i+2) / std::max(n_groundtruth, 1.0);
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

int groundtruth_class(CLASSES current_class, const tGroundtruth &groundtruth) {
  if (!strcasecmp(groundtruth.box.type.c_str(), CLASS_NAMES[current_class].c_str())) return 1;
  bool pedestrian_neighbor = !strcasecmp(CLASS_NAMES[current_class].c_str(), "Pedestrian") &&
                             !strcasecmp("Person_sitting", groundtruth.box.type.c_str());
  bool car_neighbor = !strcasecmp(CLASS_NAMES[current_class].c_str(), "Car") &&
                      !strcasecmp("Van", groundtruth.box.type.c_str());
  return pedestrian_neighbor || car_neighbor ? 0 : -1;
}

int groundtruth_ignore_code(CLASSES current_class, const tGroundtruth &groundtruth,
                            DIFFICULTY difficulty) {
  int valid_class = groundtruth_class(current_class, groundtruth);
  double height = groundtruth.box.y2 - groundtruth.box.y1;
  bool ignored = groundtruth.occlusion > MAX_OCCLUSION[difficulty] ||
                 groundtruth.truncation > MAX_TRUNCATION[difficulty] ||
                 height <= MIN_HEIGHT[difficulty];
  if (valid_class == 1 && !ignored) return 0;
  if (valid_class == 0 || (ignored && valid_class == 1)) return 1;
  return -1;
}

int detection_ignore_code(CLASSES current_class, const tDetection &detection,
                          DIFFICULTY difficulty) {
  if (fabs(detection.box.y1 - detection.box.y2) < MIN_HEIGHT[difficulty]) return 1;
  return !strcasecmp(detection.box.type.c_str(), CLASS_NAMES[current_class].c_str()) ? 0 : -1;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt,
        const vector<tDetection> &det, vector<int32_t> &ignored_gt,
        vector<tGroundtruth> &dc, vector<int32_t> &ignored_det,
        int32_t &n_gt, DIFFICULTY difficulty) {
  for (const auto &groundtruth : gt) {
    int code = groundtruth_ignore_code(current_class, groundtruth, difficulty);
    ignored_gt.push_back(code);
    if (code == 0) ++n_gt;
    if (!strcasecmp("DontCare", groundtruth.box.type.c_str())) dc.push_back(groundtruth);
  }
  for (const auto &detection : det) {
    ignored_det.push_back(detection_ignore_code(current_class, detection, difficulty));
  }
}

std::pair<int32_t, double> find_detection_candidate(
        CLASSES current_class, const tGroundtruth &groundtruth,
        const vector<tDetection> &detections, const vector<int32_t> &ignored_det,
        const vector<bool> &assigned, const vector<bool> &ignored_threshold,
        bool compute_fp, double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        METRIC metric) {
  const double no_detection = -10000000;
  int32_t selected = -1;
  double candidate_score = no_detection;
  double max_overlap = 0;
  bool selected_is_ignored = false;
  for (int32_t j = 0; j < static_cast<int32_t>(detections.size()); ++j) {
    if (ignored_det[j] == -1 || assigned[j] || ignored_threshold[j]) continue;
    double overlap = boxoverlap(detections[j], groundtruth, -1);
    bool valid_overlap = overlap > MIN_OVERLAP[metric][current_class];
    if (!compute_fp && valid_overlap && detections[j].thresh > candidate_score) {
      selected = j;
      candidate_score = detections[j].thresh;
    } else if (compute_fp && valid_overlap && ignored_det[j] == 0 &&
               (overlap > max_overlap || selected_is_ignored)) {
      max_overlap = overlap;
      selected = j;
      candidate_score = 1;
      selected_is_ignored = false;
    } else if (compute_fp && valid_overlap && candidate_score == no_detection &&
               ignored_det[j] == 1) {
      selected = j;
      candidate_score = 1;
      selected_is_ignored = true;
    }
  }
  return {selected, candidate_score};
}

void assign_groundtruth_detections(
        CLASSES current_class, const vector<tGroundtruth> &groundtruth,
        const vector<tDetection> &detections, const vector<int32_t> &ignored_gt,
        const vector<int32_t> &ignored_det, const vector<bool> &ignored_threshold,
        bool compute_fp, bool compute_aos,
        double (*boxoverlap)(tDetection, tGroundtruth, int32_t), METRIC metric,
        vector<bool> &assigned, vector<double> &delta, tPrData &stat) {
  const double no_detection = -10000000;
  for (int32_t i = 0; i < static_cast<int32_t>(groundtruth.size()); ++i) {
    if (ignored_gt[i] == -1) continue;
    auto match = find_detection_candidate(current_class, groundtruth[i], detections,
        ignored_det, assigned, ignored_threshold, compute_fp, boxoverlap, metric);
    if (match.second == no_detection && ignored_gt[i] == 0) {
      ++stat.fn;
    } else if (match.second != no_detection) {
      assigned[match.first] = true;
      if (ignored_gt[i] == 0 && ignored_det[match.first] == 0) {
        ++stat.tp;
        stat.v.push_back(detections[match.first].thresh);
        if (compute_aos) delta.push_back(
            groundtruth[i].box.alpha - detections[match.first].box.alpha);
      }
    }
  }
}

int suppress_dontcare_detections(
        CLASSES current_class, const vector<tGroundtruth> &dontcare,
        const vector<tDetection> &detections, const vector<int32_t> &ignored_det,
        const vector<bool> &ignored_threshold,
        double (*boxoverlap)(tDetection, tGroundtruth, int32_t), METRIC metric,
        vector<bool> &assigned) {
  int suppressed = 0;
  for (const auto &area : dontcare) {
    for (int32_t j = 0; j < static_cast<int32_t>(detections.size()); ++j) {
      if (assigned[j] || ignored_det[j] != 0 || ignored_threshold[j]) continue;
      if (boxoverlap(detections[j], area, 0) > MIN_OVERLAP[metric][current_class]) {
        assigned[j] = true;
        ++suppressed;
      }
    }
  }
  return suppressed;
}

void calculate_false_positives(
        CLASSES current_class, const vector<tGroundtruth> &dontcare,
        const vector<tDetection> &detections, const vector<int32_t> &ignored_det,
        const vector<bool> &ignored_threshold, bool compute_aos,
        double (*boxoverlap)(tDetection, tGroundtruth, int32_t), METRIC metric,
        vector<bool> &assigned, const vector<double> &delta, tPrData &stat) {
  for (int32_t i = 0; i < static_cast<int32_t>(detections.size()); ++i) {
    if (!assigned[i] && ignored_det[i] == 0 && !ignored_threshold[i]) ++stat.fp;
  }
  stat.fp -= suppress_dontcare_detections(current_class, dontcare, detections,
      ignored_det, ignored_threshold, boxoverlap, metric, assigned);
  if (!compute_aos) return;
  vector<double> similarities(stat.fp, 0);
  for (double angle : delta) similarities.push_back((1.0 + cos(angle)) / 2.0);
  if (similarities.size() != static_cast<size_t>(stat.fp + stat.tp))
    throw std::logic_error("orientation statistics size mismatch");
  if (delta.size() != static_cast<size_t>(stat.tp))
    throw std::logic_error("orientation delta size mismatch");
  stat.similarity = stat.tp > 0 || stat.fp > 0
      ? accumulate(similarities.begin(), similarities.end(), 0.0) : -1;
}

tPrData computeStatistics(CLASSES current_class, const vector<tGroundtruth> &gt,
        const vector<tDetection> &det, const vector<tGroundtruth> &dc,
        const vector<int32_t> &ignored_gt, const vector<int32_t> &ignored_det,
        bool compute_fp, double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        METRIC metric, bool compute_aos=false, double thresh=0, bool debug=false) {
  static_cast<void>(debug);
  tPrData stat;
  vector<bool> assigned(det.size(), false);
  vector<bool> ignored_threshold(det.size(), false);
  if (compute_fp) {
    for (int32_t i = 0; i < static_cast<int32_t>(det.size()); ++i)
      ignored_threshold[i] = det[i].thresh < thresh;
  }
  vector<double> delta;
  assign_groundtruth_detections(current_class, gt, det, ignored_gt, ignored_det,
      ignored_threshold, compute_fp, compute_aos, boxoverlap, metric,
      assigned, delta, stat);
  if (compute_fp) {
    calculate_false_positives(current_class, dc, det, ignored_det,
        ignored_threshold, compute_aos, boxoverlap, metric, assigned, delta, stat);
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

void summarize_precision(const vector<tPrData> &stats, size_t threshold_count,
        bool compute_aos, vector<double> &precision, vector<double> &aos) {
  precision.assign(N_SAMPLE_PTS, 0);
  if (compute_aos) aos.assign(N_SAMPLE_PTS, 0);
  for (size_t i = 0; i < threshold_count; ++i) {
    precision[i] = stats[i].tp / static_cast<double>(stats[i].tp + stats[i].fp);
    if (compute_aos)
      aos[i] = stats[i].similarity / static_cast<double>(stats[i].tp + stats[i].fp);
  }
  for (size_t i = 0; i < threshold_count; ++i) {
    precision[i] = *max_element(precision.begin() + i, precision.end());
    if (compute_aos) aos[i] = *max_element(aos.begin() + i, aos.end());
  }
}

bool eval_class (FILE *fp_det, FILE *fp_ori, CLASSES current_class,
        const vector< vector<tGroundtruth> > &groundtruth,
        const vector< vector<tDetection> > &detections, bool compute_aos,
        double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        vector<double> &precision, vector<double> &aos,
        DIFFICULTY difficulty, METRIC metric) {
    if (groundtruth.size() != detections.size())
      throw std::invalid_argument("ground-truth and detection counts differ");

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  // for all test images do
  for (int32_t i=0; i<groundtruth.size(); i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false, boxoverlap, metric);

    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);

  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());
  for (int32_t i=0; i<groundtruth.size(); i++){

    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, boxoverlap, metric,
                              compute_aos, thresholds[t], t==38);

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1)
        pr[t].similarity += tmp.similarity;
    }
  }

  summarize_precision(pr, thresholds.size(), compute_aos, precision, aos);

  // save statisics and finish with success
  saveStats(precision, aos, fp_det, fp_ori);
    return true;
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos){

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  printf("save %s\n", (dir_name + "/" + file_name + ".txt").c_str());
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i]);
  fclose(fp);

#ifdef OFFLINE_EVAL
  float sum[3] = {0, 0, 0};
  for (int value_index = 0; value_index < 3; ++value_index)
    for (int sample_index = 0; sample_index < vals[value_index].size(); sample_index += 4)
      sum[value_index] += vals[value_index][sample_index];
  printf("%s AP: %f %f %f\n", file_name.c_str(),
         sum[0] / 11 * 100, sum[1] / 11 * 100, sum[2] / 11 * 100);
#endif

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    string command = "cd " + dir_name + "; gnuplot " + file_name + ".gp";
    system(command.c_str());
  }

  // create pdf and crop
  string command = "cd " + dir_name + "; ps2pdf " + file_name + ".eps "
      + file_name + "_large.pdf";
  system(command.c_str());
  command = "cd " + dir_name + "; pdfcrop " + file_name + "_large.pdf "
      + file_name + ".pdf";
  system(command.c_str());
  command = "cd " + dir_name + "; rm " + file_name + "_large.pdf";
  system(command.c_str());
}

#ifdef OFFLINE_EVAL
vector<int32_t> getEvalIndices(const string& result_dir) {
  vector<int32_t> indices;
  DIR* directory = opendir(result_dir.c_str());
  if (directory == nullptr) return indices;
  dirent* entity;
  while ((entity = readdir(directory)) != nullptr) {
    string path(entity->d_name);
    int32_t length = path.size();
    if (length < 10) continue;
    indices.push_back(atoi(path.substr(length - 10, 10).c_str()));
  }
  closedir(directory);
  return indices;
}
#endif

struct EvaluationData {
  vector<vector<tGroundtruth>> groundtruth;
  vector<vector<tDetection>> detections;
  bool compute_aos;
  vector<bool> eval_image;
  vector<bool> eval_ground;
  vector<bool> eval_3d;
  EvaluationData() : compute_aos(true), eval_image(NUM_CLASS, false),
      eval_ground(NUM_CLASS, false), eval_3d(NUM_CLASS, false) {}
};

bool load_evaluation_frames(const string &gt_dir, const string &result_dir,
        const vector<int32_t> &indices, Mail *mail, EvaluationData &data) {
  mail->msg("Loading detections...");
  for (int32_t index : indices) {
    ostringstream file_name_stream;
    file_name_stream << setw(6) << setfill('0') << index << ".txt";
    string file_name = file_name_stream.str();
    bool gt_success;
    bool detection_success;
    data.groundtruth.push_back(loadGroundtruth(gt_dir + "/" + file_name, gt_success));
    data.detections.push_back(loadDetections(result_dir + "/data/" + file_name,
        data.compute_aos, data.eval_image, data.eval_ground, data.eval_3d,
        detection_success));
    if (!gt_success || !detection_success) {
      mail->msg("ERROR: Couldn't read: %s", file_name.c_str());
      return false;
    }
  }
  mail->msg("  done.");
  return true;
}

bool evaluate_metric_classes(const string &result_dir, const string &plot_dir,
        const string &suffix, const vector<bool> &enabled, bool compute_aos,
        double (*overlap)(tDetection, tGroundtruth, int32_t), METRIC metric,
        const EvaluationData &data, bool make_plots, Mail *mail) {
  for (int c = 0; c < NUM_CLASS; ++c) {
    if (!enabled[c]) continue;
    FILE *fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + suffix + ".txt").c_str(), "w");
    FILE *fp_ori = compute_aos
        ? fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_orientation.txt").c_str(), "w")
        : nullptr;
    vector<double> precision[3], aos[3];
    CLASSES current_class = static_cast<CLASSES>(c);
    bool success = eval_class(fp_det, fp_ori, current_class, data.groundtruth,
        data.detections, compute_aos, overlap, precision[0], aos[0], EASY, metric) &&
        eval_class(fp_det, fp_ori, current_class, data.groundtruth,
        data.detections, compute_aos, overlap, precision[1], aos[1], MODERATE, metric) &&
        eval_class(fp_det, fp_ori, current_class, data.groundtruth,
        data.detections, compute_aos, overlap, precision[2], aos[2], HARD, metric);
    fclose(fp_det);
    if (fp_ori != nullptr) fclose(fp_ori);
    if (!success) {
      mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
      return false;
    }
    if (make_plots) {
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + suffix, CLASS_NAMES[c], precision, false);
      if (compute_aos)
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation", CLASS_NAMES[c], aos, true);
    }
  }
  return true;
}

#ifndef OFFLINE_EVAL
bool eval(string result_sha, Mail *mail) {
  initGlobals();
  string gt_dir = "data/object/label_2";
  string result_dir = "results/" + result_sha;
  string plot_dir = result_dir + "/plot";
  system(("mkdir " + plot_dir).c_str());
  vector<int32_t> indices;
  for (int32_t i = 0; i < N_TESTIMAGES; ++i) indices.push_back(i);
  EvaluationData data;
  if (!load_evaluation_frames(gt_dir, result_dir, indices, mail, data)) return false;
  return evaluate_metric_classes(result_dir, plot_dir, "_detection", data.eval_image,
             data.compute_aos, imageBoxOverlap, IMAGE, data, false, mail) &&
         evaluate_metric_classes(result_dir, plot_dir, "_detection_ground", data.eval_ground,
             false, groundBoxOverlap, GROUND, data, false, mail) &&
         evaluate_metric_classes(result_dir, plot_dir, "_detection_3d", data.eval_3d,
             false, box3DOverlap, BOX3D, data, false, mail);
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=4) {
    cout << "Usage: ./eval_detection result_sha [user_sha email]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];

  // init notification mail
  Mail *mail;
  if (argc==4) mail = new Mail(argv[3]);
  else         mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  if (eval(result_sha,mail)) {
    mail->msg("Your evaluation results are available at:");
    mail->msg("http://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=object&user=%s&result=%s",argv[2], result_sha.c_str());
  } else {
    system(("rm -r results/" + result_sha).c_str());
    mail->msg("An error occured while processing your results.");
    mail->msg("Please make sure that the data in your zip archive has the right format!");
  }

  // send mail and exit
  delete mail;

  return 0;
}
#else
bool eval(string gt_dir, string result_dir, Mail *mail) {
  initGlobals();
  string plot_dir = result_dir + "/plot";
  system(("mkdir " + plot_dir).c_str());
  vector<int32_t> indices = getEvalIndices(result_dir + "/data/");
  printf("number of files for evaluation: %d\n", static_cast<int>(indices.size()));
  EvaluationData data;
  if (!load_evaluation_frames(gt_dir, result_dir, indices, mail, data)) return false;
  bool image_success = evaluate_metric_classes(result_dir, plot_dir, "_detection",
      data.eval_image, data.compute_aos, imageBoxOverlap, IMAGE, data, true, mail);
  printf("Finished 2D bounding box eval.\n");
  bool ground_success = evaluate_metric_classes(result_dir, plot_dir, "_detection_ground",
      data.eval_ground, false, groundBoxOverlap, GROUND, data, true, mail);
  printf("Finished Birdeye eval.\n");
  bool box_success = evaluate_metric_classes(result_dir, plot_dir, "_detection_3d",
      data.eval_3d, false, box3DOverlap, BOX3D, data, true, mail);
  printf("Finished 3D bounding box eval.\n");
  return image_success && ground_success && box_success;
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc != 3) {
    cout << "Usage: ./eval_detection_3d_offline gt_dir result_dir" << endl;
    return 1;
  }
  string gt_dir = argv[1];
  string result_dir = argv[2];
  Mail *mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");
  if (eval(gt_dir, result_dir, mail)) {
    mail->msg("Your evaluation results are available at:");
    mail->msg(result_dir.c_str());
  } else {
    system(("rm -r " + result_dir + "/plot").c_str());
    mail->msg("An error occured while processing your results.");
  }
  delete mail;
  return 0;
}
#endif
