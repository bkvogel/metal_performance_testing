#ifndef _ASSERTIONS_H
#define _ASSERTIONS_H
/*
 * Copyright (c) 2005-2015, Brian K. Vogel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 *
 */

#include <iostream>




/**
 * Print an error message and exit.
 *
 * When debugging, set a breakpoint inside this function to view the stack trace.
 *
 * @param reason Reason for error.
 */
void error_exit(const std::string& reason);

/**
 * Assert that a condition is expected to be true and abort with an error message
 * otherwise.
 *
 * Implementation note: This function calls "error_exit()." To get a stack trace when
 * an assertion fails, set a break point inside "error_exit()".
 *
 * @param condition A condition that is expected to evalute to true. If it evaluates
 * to false, \p message will be printed to stdout and the program will then abort.
 *
 * @param message An error message to print if the assertion check fails. If not specified,
 * an uninformative default error message will be used instead.
 */
void assertion(bool condition, const std::string& message="Error: the type of error that just occured is not worth mentioning.");




#endif  /* _ASSERTIONS_H */
