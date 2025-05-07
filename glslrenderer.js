/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
*/

"use strict";

class GlslRenderer
{
    constructor(canvas)
    {
        this.canvas = canvas;
        this.renderCallback = null;

        this.gl = canvas.getContext("webgl2");
        if (!this.gl) {
            console.error("Failed to create a WebGL2 context");
            return;
        }

        this.finalRenderTarget = new RenderTarget(this, this.gl, false);
        this.renderTargets = [];

        this.libraries = new Map();

        this.geometrybuffer = null;

        this.framerate = {
            frameCounter: 0,
            start: 0.0,
            previous: 0.0,
            framerate: 0.0,
            element: null,
        };

        this.frameCounter = 0;
        this.startTimestamp = 0.0;

        this.mouse = {
            current: [0.0, 0.0],
            click: [-1.0, -1.0],
            pressed: false,
        };

        // Enable linear filtering for floating-point textures if possible
        this.gl.linearFilteringAvailable = (this.gl.getExtension('OES_texture_float_linear') != null);
        this.gl.renderToFloat32FAvailable = (this.gl.getExtension( 'EXT_color_buffer_float') != null);

        this.canvas.addEventListener("mousedown", (event) => {
            if ((event.button == 0) && !event.altKey && !event.ctrlKey && !event.shiftKey && !event.metaKey)
            {
                this.mouse.click = [event.x, this.canvas.height - event.y];
                this.mouse.current = this.mouse.click;
                this.mouse.pressed = true;
            }
        });

        this.canvas.addEventListener("mouseup", (event) => {
            if (this.mouse.pressed && (event.button == 0))
            {
                this.mouse.click = [-1.0, -1.0];
                this.mouse.pressed = false;
            }
        });

        this.canvas.addEventListener("mousemove", (event) => {
            if (this.mouse.pressed)
                this.mouse.current = [event.x, this.canvas.height - event.y];
        });

        this._createGeometry();
        this._render(0.0);
    }

    createRenderTarget(height=-1, width=-1)
    {
        if (height < 0)
            height = this.gl.canvas.height;

        if (width < 0)
            width = this.gl.canvas.width;

        let renderTarget = new RenderTarget(this, this.gl, true, height, width);
        this.renderTargets.push(renderTarget);
        return renderTarget;
    }

    addLibrary(name, source)
    {
        this.libraries.set(name, source);
    }

    setup(fragmentShaderSource, declarations={})
    {
        this.finalRenderTarget.setup(fragmentShaderSource, declarations);
    }

    setUniform(name, value)
    {
        this.finalRenderTarget.setUniform(name, value);
    }

    setMatrix(name, matrix, height, width, asVec4=false, useFiltering=false)
    {
        this.finalRenderTarget.setMatrix(name, matrix, height, width, asVec4, useFiltering);
    }

    setTexture(name, data, height, width, nbChannels=3, useFiltering=true)
    {
        this.finalRenderTarget.setTexture(name, data, height, width, nbChannels, useFiltering);
    }

    loadTexture(name, url, useFiltering=true)
    {
        this.finalRenderTarget.loadTexture(name, url, useFiltering);
    }

    setTextureFromRenderTarget(name, renderTarget)
    {
        this.finalRenderTarget.setTextureFromRenderTarget(name, renderTarget);
    }

    setFramerateTarget(element)
    {
        this.framerate.element = element;
    }

    setRenderCallback(callback)
    {
        this.renderCallback = callback;
    }

    _createGeometry()
    {
        this.geometrybuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.geometrybuffer);

        var positions = [
            -1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0,
            -1.0, -1.0,
            1.0, 1.0,
            1.0, -1.0,
        ];

        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);
    }

    _render(timestamp)
    {
        if (this.frameCounter == 0)
            this.startTimestamp = timestamp;

        timestamp -= this.startTimestamp;

        requestAnimationFrame((timestamp) => this._render(timestamp));

        ++this.framerate.frameCounter;
        if (this.framerate.frameCounter == 10)
        {
            this.framerate.framerate = this.framerate.frameCounter * 1000 / (timestamp - this.framerate.start);
            this.framerate.start = timestamp;
            this.framerate.frameCounter = 0;

            if (this.framerate.element != null)
                this.framerate.element.innerText = this.framerate.framerate.toFixed(0) + ' fps';
        }

        let uniforms = {
            frame: this.frameCounter,
            resolution: [this.canvas.width, this.canvas.height],
            time: timestamp / 1000.0,
            timeDelta: (timestamp - this.framerate.previous) / 1000.0,
            mouse: [this.mouse.current[0], this.mouse.current[1], this.mouse.click[0], this.mouse.click[1]],
        };

        for (let renderTarget of this.renderTargets)
            renderTarget.render(uniforms, this.geometrybuffer);

        this.finalRenderTarget.render(uniforms, this.geometrybuffer);

        if (this.renderCallback != null)
            this.renderCallback(uniforms);

        ++this.frameCounter;
        this.framerate.previous = timestamp;
    }

    _reset()
    {
        this.framerate = {
            frameCounter: 0,
            start: 0.0,
            previous: 0.0,
            framerate: 0.0,
            element: this.framerate.element,
        };

        this.frameCounter = 0;
    }
};


class Buffer
{
    constructor(gl, height, width)
    {
        this.framebuffers = [null, null];
        this.textures = [null, null];
        this.current = 1;
        this.height = height;
        this.width = width;

        for (var i = 0; i < 2; ++i)
        {
            // Create the texture
            this.textures[i] = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);

            if (gl.renderToFloat32FAvailable)
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
            else
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

            // Create the framebuffer
            this.framebuffers[i] = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[i]);

            // Attach the texture as the first color attachment
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[i], 0);

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }
    }
};


class Input
{
    location = null;
    texture = null;
    buffer = null;
};


class RenderTarget
{
    constructor(renderer, gl, useFrameBuffer, height, width)
    {
        this.renderer = renderer;
        this.gl = gl;
        this.program = null;

        this.uniforms = {
            frame: null,
            resolution: null,
            time: null,
            timeDelta: null,
            mouse: null,
            custom: new Map(),
        };

        this.customUniforms = new Map();

        this.geometryLocation = null;

        this.inputs = {};

        // Only for render-to-texture
        this.outputBuffer = null;

        if (useFrameBuffer)
            this.outputBuffer = new Buffer(this.gl, height, width);
    }

    setup(fragmentShaderSource, declarations={})
    {
        function _insertText(text, offset)
        {
            fragmentShaderSource = fragmentShaderSource.substring(0, offset) + text + fragmentShaderSource.substring(offset);
            return offset + text.length;
        }

        // Delete the program if we already use one
        if (this.program != null)
        {
            this.gl.deleteProgram(this.program);
            this.program = null;
        }

        // The vertex shader we'll use
        var vertexShaderSource = `#version 300 es

        in vec4 a_position;

        void main() {
            gl_Position = a_position;
        }`;

        // Process the fragment shader source code
        if (!fragmentShaderSource.startsWith('#version 300 es\n'))
            fragmentShaderSource = '#version 300 es\n\n' + fragmentShaderSource;

        var offset = Math.max(
            fragmentShaderSource.lastIndexOf('precision highp '),
            fragmentShaderSource.lastIndexOf('precision mediump '),
            fragmentShaderSource.lastIndexOf('precision lowp ')
        );

        if (offset > 0)
            offset = fragmentShaderSource.indexOf(';', offset) + 2;
        else
            offset = _insertText('precision mediump float;\n', 16);

        offset = _insertText('\n', offset);

        if (!(declarations instanceof Map))
            declarations = new Map(Object.entries(declarations));

        for (const [key, value] of declarations)
        {
            const declaration = '#define ' + key + ' ' + value + '\n';
            offset = _insertText(declaration, offset);
        }

        const declaration = `
            uniform vec2      iResolution;           // viewport resolution (in pixels)
            uniform float     iTime;                 // shader playback time (in seconds)
            uniform float     iTimeDelta;            // render time (in seconds)
            uniform float     iFrameRate;            // shader frame rate
            uniform int       iFrame;                // shader playback frame
            uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
            uniform sampler2D iChannel0;
            uniform sampler2D iChannel1;
            uniform sampler2D iChannel2;
            uniform sampler2D iChannel3;
        `;

        offset = _insertText(declaration, offset);

        // Insert the libraries
        offset = fragmentShaderSource.indexOf('#include');
        while (offset != -1)
        {
            const end = fragmentShaderSource.indexOf('\n', offset);
            let line = fragmentShaderSource.substring(offset, end);
            let library = line.replace('#include ', '');

            let code = '';
            if (this.renderer.libraries.has(library))
                code = this.renderer.libraries.get(library);

            fragmentShaderSource = fragmentShaderSource.substring(0, offset) + code + fragmentShaderSource.substring(end);

            offset = fragmentShaderSource.indexOf('#include');
        }

        // Add the main function
        fragmentShaderSource += `

        out vec4 fragColor;

        void main() {
            mainImage(fragColor, gl_FragCoord.xy);
        }`;

        // Compile the shaders
        var vertexShader = _createShader(this.gl, this.gl.VERTEX_SHADER, vertexShaderSource);
        var fragmentShader = _createShader(this.gl, this.gl.FRAGMENT_SHADER, fragmentShaderSource);

        this.program = _createProgram(this.gl, vertexShader, fragmentShader);

        // Retrieve the location of the uniforms
        this.uniforms.resolution = this.gl.getUniformLocation(this.program, "iResolution");
        this.uniforms.time = this.gl.getUniformLocation(this.program, "iTime");
        this.uniforms.timeDelta = this.gl.getUniformLocation(this.program, "iTimeDelta");
        this.uniforms.frame = this.gl.getUniformLocation(this.program, "iFrame");
        this.uniforms.mouse = this.gl.getUniformLocation(this.program, "iMouse");

        for (const [key, value] of this.customUniforms)
            this.uniforms.custom.set(key, this.gl.getUniformLocation(this.program, key));

        for (var key in this.inputs)
        {
            const input = this.inputs[key];
            input.location = this.gl.getUniformLocation(this.program, key);
        }

        // Retrieve the location of the position buffer
        this.geometryLocation = this.gl.getAttribLocation(this.program, "a_position");

        this.renderer._reset();
    }

    setUniform(name, value)
    {
        this.customUniforms.set(name, value);
    }

    setMatrix(name, matrix, height, width, asVec4=false, useFiltering=false)
    {
        let filtering = (useFiltering && this.gl.linearFilteringAvailable ? this.gl.LINEAR : this.gl.NEAREST);

        this._createOrActivateTexture(name, filtering);

        if (Array.isArray(matrix))
            matrix = new Float32Array(matrix.flat(Infinity));

        if (asVec4)
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA32F, width / 4, height, 0, this.gl.RGBA, this.gl.FLOAT, matrix);
        else
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, width, height, 0, this.gl.RED, this.gl.FLOAT, matrix);
    }

    setTexture(name, data, height, width, nbChannels=3, useFiltering=true)
    {
        let filtering = (useFiltering ? this.gl.LINEAR : this.gl.NEAREST);

        this._createOrActivateTexture(name, filtering);

        if (Array.isArray(data))
            data = new UInt8Array(matrix.flat(Infinity));

        if (nbChannels == 4)
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, data);
        else if (nbChannels == 3)
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGB8, width, height, 0, this.gl.RGB, this.gl.UNSIGNED_BYTE, data);
        else if (nbChannels == 2)
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RG8, width, height, 0, this.gl.RG, this.gl.UNSIGNED_BYTE, data);
        else
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R8, width, height, 0, this.gl.R8, this.gl.UNSIGNED_BYTE, data);

        this.renderer._reset();
    }

    loadTexture(name, url, useFiltering=true)
    {
        const renderTarget = this;

        var image = new Image();
        image.onload = function () {
            renderTarget.setTexture(name, this, this.height, this.width, 3, useFiltering);
        }
        image.src = url;
    }

    setTextureFromRenderTarget(name, renderTarget)
    {
        if (this.inputs[name] == undefined)
            this.inputs[name] = new Input();

        let input = this.inputs[name];
        input.buffer = renderTarget.outputBuffer;
        input.texture = null;

        if (this.program != null)
            input.location = this.gl.getUniformLocation(this.program, name);

        this.renderer._reset();
    }

    render(uniforms, geometryBuffer)
    {
        // Activate the framebuffer, either the one from the canvas or from the render texture
        if (this.outputBuffer != null)
        {
            let dst = 1 - this.outputBuffer.current;
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.outputBuffer.framebuffers[dst]);
        }
        else
        {
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        }

        if (this.outputBuffer != null)
            this.gl.viewport(0, 0, this.outputBuffer.width, this.outputBuffer.height);
        else
            this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

        // Clear the canvas
        if (uniforms.frame == 0)
        {
            this.gl.clearColor(0, 0, 0, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        }

        if (this.program == null)
            return;

        // Tell it to use our program
        this.gl.useProgram(this.program);

        // Send the uniforms to the shader
        this.gl.uniform1i(this.uniforms.frame, uniforms.frame);
        this.gl.uniform2f(this.uniforms.resolution, uniforms.resolution[0], uniforms.resolution[1]);
        this.gl.uniform1f(this.uniforms.time, uniforms.time);
        this.gl.uniform1f(this.uniforms.timeDelta, uniforms.timeDelta);
        this.gl.uniform4f(this.uniforms.mouse, uniforms.mouse[0], uniforms.mouse[1], uniforms.mouse[2], uniforms.mouse[3]);

        for (const [key, value] of this.customUniforms)
        {
            if (Array.isArray(value))
            {
                if (value.length == 4)
                    this.gl.uniform4f(this.uniforms.custom.get(key), value[0], value[1], value[2], value[3]);
                else if (value.length == 3)
                    this.gl.uniform3f(this.uniforms.custom.get(key), value[0], value[1], value[2]);
                else if (value.length == 2)
                    this.gl.uniform2f(this.uniforms.custom.get(key), value[0], value[1]);
                else
                    this.gl.uniform1f(this.uniforms.custom.get(key), value[0]);
            }
            else
            {
                this.gl.uniform1f(this.uniforms.custom.get(key), value);
            }
        }

        // Send the textures to the shader
        var index = 0;
        for (var key in this.inputs)
        {
            const input = this.inputs[key];

            this.gl.activeTexture(this.gl.TEXTURE0 + index);

            if (input.buffer != null)
                this.gl.bindTexture(this.gl.TEXTURE_2D, input.buffer.textures[input.buffer.current]);
            else
                this.gl.bindTexture(this.gl.TEXTURE_2D, input.texture);

            this.gl.uniform1i(input.location, index);

            ++index;
        }

        // Send the buffers needed by geometry
        this.gl.enableVertexAttribArray(this.geometryLocation);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, geometryBuffer);
        this.gl.vertexAttribPointer(this.geometryLocation, 2, this.gl.FLOAT, false, 0, 0);

        // Draw the geometry
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

        // Cleanup
        if (this.outputBuffer != null)
        {
            this.outputBuffer.current = 1 - this.outputBuffer.current;
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        }

        for (var i = 0; i < this.inputs.length; ++i)
        {
            this.gl.activeTexture(this.gl.TEXTURE0 + i);
            this.gl.bindTexture(this.gl.TEXTURE_2D, null);
        }
    }

    _createOrActivateTexture(name, filtering)
    {
        if (this.inputs[name] == undefined)
        {
            let input = new Input();

            if (this.program != null)
                input.location = this.gl.getUniformLocation(this.program, name);

            this.gl.activeTexture(this.gl.TEXTURE0);

            input.texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, input.texture);

            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, filtering);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, filtering);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.REPEAT);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.REPEAT);

            this.inputs[name] = input;
        }
        else
        {
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.inputs[name].texture);
        }
    }
};


function _createProgram(gl, vertexShader, fragmentShader)
{
    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    var success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success)
        return program;

    console.log(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
}


function _createShader(gl, type, source)
{
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success)
        return shader;

    console.log(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
}


globalThis.GlslRenderer = GlslRenderer;
