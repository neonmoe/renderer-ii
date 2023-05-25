#version 450 core

void main() {
    // Outputs triangles that cover the entire viewport for rendering post-processing effects.
    // Depth tests and writes should be turned off.
    int index = gl_VertexIndex % 3;
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    } else if (index == 1) {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
    } else {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
    }
}
