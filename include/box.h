#ifndef BOX_H
#define BOX_H

#include "rtweekend.h"

#include "aabb.h"
#include "aarect.h"
#include "hittable_list.h"

class box : public hittable
{
public:
    box() {}
    box(const point3 &p0, const point3 &p1, shared_ptr<material> ptr);

    virtual bool hit(const ray &r, double t0, double t1, hit_record &rec) const override;
    virtual bool bounding_box(double t0, double t1, aabb &output_box) const override;

public:
    point3 box_min;
    point3 box_max;
    hittable_list sides;
};

#endif