/* -*- c++ -*- */
/*
 * Copyright 2025 Trevor, Skylar, Jorge, Kobe.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUSTOMMODULE_CORSYMIN_H
#define INCLUDED_CUSTOMMODULE_CORSYMIN_H

#include <gnuradio/customModule/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace customModule {

/*!
 * \brief <+description of block+>
 * \ingroup customModule
 *
 */
class CUSTOMMODULE_API corSymIn : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<corSymIn> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of customModule::corSymIn.
     *
     * To avoid accidental use of raw pointers, customModule::corSymIn's
     * constructor is in a private implementation
     * class. customModule::corSymIn::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace customModule
} // namespace gr

#endif /* INCLUDED_CUSTOMMODULE_CORSYMIN_H */
