/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QString>
#include <QColor>

// Convert a std::string to a QString
QString inline ss2qs(const std::string& str)
{
	QString sResult =  QString::fromUtf8( str.c_str() );
	sResult.replace( "mkm", QString(QChar(0x00b5))+"m" );
	return sResult;
}

// Convert a QString to a std::string
std::string inline qs2ss(const QString& str)
{
	return str.toStdString();
}

// color of the table element which cannot be changed
QColor inline InactiveTableColor()
{
	return QColor::fromRgb(150, 150, 150);
}

// add QOverload to older Qt versions (is used for linux)
#if QT_VERSION < QT_VERSION_CHECK(5, 7, 0)
#if defined(Q_COMPILER_VARIADIC_TEMPLATES)
template <typename... Args> struct QNonConstOverload
{
	template <typename R, typename T> Q_DECL_CONSTEXPR auto operator()(R(T::*ptr)(Args...)) const Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
	template <typename R, typename T> static Q_DECL_CONSTEXPR auto of(R(T::*ptr)(Args...)) Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
};
template <typename... Args> struct QConstOverload
{
	template <typename R, typename T> Q_DECL_CONSTEXPR auto operator()(R(T::*ptr)(Args...) const) const Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
	template <typename R, typename T> static Q_DECL_CONSTEXPR auto of(R(T::*ptr)(Args...) const) Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
};
template <typename... Args> struct QOverload : QConstOverload<Args...>, QNonConstOverload<Args...>
{
	using QConstOverload<Args...>::of;
	using QConstOverload<Args...>::operator();
	using QNonConstOverload<Args...>::of;
	using QNonConstOverload<Args...>::operator();
	template <typename R> Q_DECL_CONSTEXPR auto operator()(R(*ptr)(Args...)) const Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
	template <typename R> static Q_DECL_CONSTEXPR auto of(R(*ptr)(Args...)) Q_DECL_NOTHROW -> decltype(ptr) { return ptr; }
};
#if defined(__cpp_variable_templates) && __cpp_variable_templates >= 201304 // C++14
template <typename... Args> Q_CONSTEXPR Q_DECL_UNUSED QOverload<Args...> qOverload = {};
template <typename... Args> Q_CONSTEXPR Q_DECL_UNUSED QConstOverload<Args...> qConstOverload = {};
template <typename... Args> Q_CONSTEXPR Q_DECL_UNUSED QNonConstOverload<Args...> qNonConstOverload = {};
#endif
#endif
#endif